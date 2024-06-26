import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from datetime import datetime
from work_mem_transformer import Transformer, CustomSchedule, create_masks
from utils import load_data


def nucleus_sampling(x):
	logits = x[0] 
	p = x[1]
	# implementation from https://github.com/royRLL/Zen-NLG-using-Tensorflow-and-Nucleus-Sampling/blob/main/ProjectCharacterPredictionHP60.ipynb
	sortedLogits = tf.sort(logits, direction='DESCENDING')
	sortedProbs = tf.nn.softmax(sortedLogits)
	probsSum = tf.cumsum(sortedProbs, exclusive=True)
	maskedLogits = tf.where(probsSum < p, sortedLogits,
							tf.ones_like(sortedLogits, dtype=tf.float32) * 1e9)  
	minLogits = tf.reduce_min(maskedLogits, keepdims=True)  
	res_logits = tf.where(
			logits < minLogits,
			tf.ones_like(logits, dtype=tf.float32) * -1e9,
			logits)
	sample = tf.random.categorical(tf.expand_dims(res_logits, 0), 1)
	return sample[0]

def sample_tokens(x):
	preds = x[0]
	token_type = x[1]
	nucleus_p = x[2]
	return tf.cond(tf.equal(token_type, 0),
				   true_fn=lambda: nucleus_sampling((tf.squeeze(preds, [0]), nucleus_p)),
				   false_fn=lambda: tf.argmax(preds, axis=-1))


def train(config):
	"""
	example config dict
	config = {
	"num_layers": 4, 
	"d_model": 128, 
	"dff": 512, 
	"num_heads": 8,
	"dropout_rate": 0.1,
	"token_type": True,
	"beta_1": 0.9,
	"beta_2": 0.98,
	"epsilon": 1e-9,
	"data_path": 'ted_hrlr_translate/pt_to_en',
	"tokenizer_path": '../wmt14/tokenizers/pt-en',
	"max_length": 40,
	"buffer_size": 20000,
	"batch_size": 64,
	"mem_tokens_num": 10,
	"nucleus_p": 0.9,
	"checkpoint_path": './ckpts_baseline+work_mem_nucleus_0.9',
	"epochs_pretrain": 5,
	"epochs_with_mem": 15
	}
	"""
	train_dataset, tokenizer_inp, tokenizer_tar = load_data(
		data_path=config['data_path'], tokenizer_path=config['tokenizer_path'],
		max_length=config['max_length'])
	train_dataset = train_dataset.shuffle(config['buffer_size']).padded_batch(config['batch_size'])
	
	strategy = tf.distribute.MirroredStrategy()
	train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
	
	input_vocab_size = tokenizer_inp.vocab_size + 2
	target_vocab_size = tokenizer_tar.vocab_size + 2
		
	learning_rate = CustomSchedule(config['d_model'])
	optimizer = tf.keras.optimizers.Adam(learning_rate, 
										 beta_1=config['beta_1'],
										 beta_2=config['beta_2'],
										 epsilon=config['epsilon']
										)
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=True, reduction='none')
	
	def loss_function(real, pred, global_batch_num_elems):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		# loss per example in global batch
		return tf.reduce_sum(loss_)/(global_batch_num_elems)
	
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
	
	transformer = Transformer(config['num_layers'], config['d_model'], 
							  config['num_heads'], config['dff'],
							  input_vocab_size, target_vocab_size,
							  pe_input=input_vocab_size,
							  pe_target=target_vocab_size,
							  rate=config['dropout_rate'],
							  token_type=config['token_type'])
	
	
	def pretrain_step(inputs, global_batch_num_elems): 
		inp, tar = inputs
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
	
		with tf.GradientTape() as tape:
			enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
			predictions = transformer(inp, tar_inp, True, enc_padding_mask,
									  combined_mask, dec_padding_mask,
									  token_type_inp=tf.ones(tf.shape(tar_inp), dtype=tf.int64))[0]
			token_id_preds = predictions[:, :, :-2]    
			loss = loss_function(tar_real, token_id_preds, global_batch_num_elems)
		gradients = tape.gradient(loss, transformer.trainable_variables)    
		optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
		train_loss(loss)
		train_accuracy(tar_real, token_id_preds)
		return loss 

	@tf.function(input_signature=[train_dist_dataset.element_spec])
	def distributed_pretrain_step(dataset_inputs):
		distr_tar = dataset_inputs[1]
		concat_tar = tf.concat(distr_tar.values, axis=0)[:, 1:]
		global_batch_num_elems = tf.reduce_sum(
			tf.cast(tf.math.logical_not(tf.math.equal(concat_tar, 0)),
					dtype=tf.float32))
		per_replica_losses = strategy.run(pretrain_step, 
										  args=(dataset_inputs, global_batch_num_elems,))
		return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


	ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, config['checkpoint_path'], max_to_keep=None)
	if not os.path.exists(config['checkpoint_path']):
		os.makedirs(config['checkpoint_path'])
	log_file = os.path.join(checkpoint_path, 'log.txt')
	
	start_epoch = 0
	# skip training without memory if the respective checkpoint exists
	if os.path.exists(os.path.join(checkpoint_path, f"ckpt-{config['epochs_pretrain']}")):
		start_epoch = config["epochs_pretrain"]
	
	# pre-training vanilla Transformer
	for epoch in range(start_epoch, config["epochs_pretrain"]):
		# train loop
		total_loss = 0.0
		num_batches = 0
		for x in tqdm(train_dist_dataset):
			total_loss += distributed_pretrain_step(x)
			num_batches += 1
			if num_batches % 50 == 0:
				with open(log_file, 'a') as f: 
					f.write(
						f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}, Batch {num_batches}, Loss: {total_loss / num_batches}, distr_elem_loss {train_loss.result()},  Accuracy: {train_accuracy.result()}\n"
						)
		train_loss_averaged = total_loss / num_batches
		ckpt_save_path = ckpt_manager.save()
		with open(log_file, 'a') as f: 
			f.write(
				f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}, Loss: {train_loss_averaged}, distr_elem_loss {train_loss.result()},  Accuracy: {train_accuracy.result()}, Checkpoint: {ckpt_save_path}\n"
			)
		train_loss.reset_states()
		train_accuracy.reset_states()


	# preparing functions for fine-tuning with memory
	def sample_teacher_forcing(x):
		predicted_id_sample = x[0]
		token_type_sample = x[1] 
		tar_inp_sample = x[2] 
		tar_real_sample = x[3]
		
		#idx discards mem token types and start token type
		idx = tf.math.reduce_sum(token_type_sample[1:-1])
		token_type_sample = tf.cond(tf.math.greater(
			tf.math.subtract(
				tf.cast(tf.shape(token_type_sample[1:])[0], dtype=tf.int64),
				tf.math.reduce_sum(token_type_sample[1:])),
			config['mem_tokens_num']),
									true_fn=lambda : tf.concat(
										[token_type_sample[:-1], tf.constant([1], dtype=tf.int64)], 0),
									false_fn=lambda : token_type_sample
								   )
		tar_inp_sample = tf.cond(tf.math.logical_and(
			tf.equal(token_type_sample[-1], 1),
			tf.less(tf.cast(idx, dtype=tf.int32),
					tf.shape(tar_real_sample)[0])
		),
								 true_fn=lambda : tf.concat([
									 tar_inp_sample, [tar_real_sample[idx]]], 0),
								 false_fn=lambda : tf.concat([
									 tar_inp_sample, predicted_id_sample], 0))
		return tar_inp_sample, token_type_sample

	def forward_pass(inp, tar_inp, tar_real, token_type, predictions):
		enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
		predictions = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask,
								  dec_padding_mask, token_type_inp=token_type)[0]
		token_id_preds = predictions[:, :, :-2]
		predicted_token_type = tf.cast(tf.argmax(predictions[: , -1:, -2:], axis=-1), tf.int64)
		token_type = tf.concat([token_type, predicted_token_type], axis=-1)
		predicted_id = tf.cast(tf.map_fn(sample_tokens, 
										 (token_id_preds[:,-1:,:], predicted_token_type,
										  tf.ones(tf.shape(token_id_preds)[0]) * config['nucleus_p']),
										 fn_output_signature=tf.TensorSpec(
											 shape=(None), dtype=tf.int64)
										), tf.int64)
		tar_inp, token_type = tf.map_fn(sample_teacher_forcing,
										(predicted_id, token_type, tar_inp, tar_real),
										fn_output_signature=(
											tf.TensorSpec(shape=(None), dtype=tf.int64),
											tf.TensorSpec(shape=(None), dtype=tf.int64)
										))
		return inp, tar_inp, tar_real, token_type, token_id_preds
	
	forward_condition = lambda inp, tar_inp, tar_real, token_type, predictions: tf.shape(tar_inp)[1] <= tf.shape(tar_real)[1] + config['mem_tokens_num'] - 1

	def filter_sample_seq_tokens(x):
		predicted_logits_sample = x[0]
		token_type_sample = x[1]
		tar_real = x[2]
		tar_seq_len = tf.shape(tar_real)[0]
		curr_token_logits = tf.boolean_mask(predicted_logits_sample, token_type_sample)[:tar_seq_len, :]
		end_token_logits = tf.concat([
			tf.zeros(
				(tar_seq_len - tf.shape(curr_token_logits)[0], target_vocab_size - 1),
				dtype=tf.float32
			),
			tf.ones(
				(tar_seq_len - tf.shape(curr_token_logits)[0], 1),
				dtype=tf.float32)
			], -1)
		return tf.concat([curr_token_logits, end_token_logits], 0)

	def work_mem_step(inputs, global_batch_num_elems):
		inp, tar = inputs
		tar_inp = tar[:, :1] # start token with type 1 (seq)
		tar_real = tar[:, 1:]
		with tf.GradientTape() as tape:
			init_token_type = tf.concat([
				tf.zeros(tf.shape(tar_inp[:, :-1]), dtype=tf.int64),
				tf.ones(tf.shape(tar_inp[:, -1:]),dtype=tf.int64)
				], axis=-1)
			predictions = tf.zeros((tf.shape(tar_inp)[0],
									tf.shape(tar_inp)[1],
									target_vocab_size),
								   dtype=tf.float32)
			inp, tar_inp, tar_real, token_type, predictions = tf.while_loop(
				forward_condition, forward_pass,
				[inp, tar_inp, tar_real, init_token_type, predictions])
			# the last forward pass before the loss calculation to get logits for the whole sequence
			enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
			predictions = transformer(inp, tar_inp, True, enc_padding_mask,
									  combined_mask, dec_padding_mask,
									  token_type_inp=token_type)[0]
			token_id_preds = predictions[:, :, :-2]
			predicted_token_type = tf.argmax(predictions[:, -1:, -2:], 
											 axis=-1, output_type=tf.int64)
			token_type = tf.concat([token_type, predicted_token_type], axis=-1)
		
			#nucleus sampling for memory positions
			predicted_id = tf.map_fn(sample_tokens, 
									 (token_id_preds[:, -1:, :],
									  predicted_token_type,
									  tf.ones(tf.shape(token_id_preds)[0]) * config['nucleus_p']
									  ),
									 fn_output_signature=tf.TensorSpec(shape=(None), dtype=tf.int64)
									 )
			tar_inp, token_type = tf.map_fn(sample_teacher_forcing,
											(predicted_id, token_type, tar_inp, tar_real),
											fn_output_signature=(
												tf.TensorSpec(shape=(None), dtype=tf.int64),
												tf.TensorSpec(shape=(None), dtype=tf.int64))
											)
			seq_predictions = tf.map_fn(filter_sample_seq_tokens,
										(token_id_preds, token_type[:, 1:], tar_real),
										fn_output_signature=tf.TensorSpec(
											shape=(None,None), dtype=tf.float32)
										)
			loss = loss_function(tar_real, seq_predictions, global_batch_num_elems)
		gradients = tape.gradient(loss, transformer.trainable_variables)    
		optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
		train_loss(loss)
		train_accuracy(tar_real, seq_predictions)
		return loss 

	@tf.function(input_signature=[train_dist_dataset.element_spec])
	def distributed_train_step(dataset_inputs):
		distr_tar = dataset_inputs[1]
		concat_tar = tf.concat(distr_tar.values,axis=0)[:, 1:]
		global_batch_num_elems = tf.reduce_sum(
			tf.cast(tf.math.logical_not(tf.math.equal(concat_tar, 0)),
					dtype=tf.float32))
		per_replica_losses = strategy.run(work_mem_step, 
										  args=(dataset_inputs, 
												global_batch_num_elems,))
		return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
	
	# before tuning ensure we loaded the correct checkpoint after pre-training
	ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, config['checkpoint_path'], max_to_keep=None) 
	ckpt.restore(os.path.join(checkpoint_path, 
							  f"ckpt-{config['epochs_pretrain']}")
				).assert_existing_objects_matched()
	
	# fine-tuning with working memory loop 
	for epoch in range(config["epochs_pretrain"], config["epochs_with_mem"]):
		total_loss = 0.0
		num_batches = 0
		for x in tqdm(train_dist_dataset):
			total_loss += distributed_train_step(x)
			num_batches += 1
			if num_batches % 50 == 0:
				with open(log_file, 'a') as f: 
					f.write(
						f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}, Batch {num_batches}, Loss: {total_loss / num_batches}, distr_elem_loss {train_loss.result()},  Accuracy: {train_accuracy.result()}\n"
					)
		train_loss_averaged = total_loss / num_batches
		ckpt_save_path = ckpt_manager.save()
		with open(log_file, 'a') as f: 
			f.write(
				f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}, Loss: {train_loss_averaged}, distr_elem_loss {train_loss.result()},  Accuracy: {train_accuracy.result()}, Checkpoint: {ckpt_save_path}\n"
				)
		train_loss.reset_states()
		train_accuracy.reset_states()
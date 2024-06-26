import os
import time
import collections
import math
import sys
import nltk
nltk.download("wordnet")
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from datetime import datetime
from nltk.translate.meteor_score import single_meteor_score
from work_mem_transformer import Transformer, create_masks
from utils import load_data

def generate(inp_sentence, tokenizer_inp, tokenizer_tar, 
			 max_length, mem_tokens_num, raw_tokens=False):
	start_token = [tokenizer_inp.vocab_size]
	end_token = [tokenizer_inp.vocab_size + 1]
	
	if not raw_tokens:
		inp_sentence = start_token + tokenizer_inp.encode(inp_sentence) + end_token
		encoder_input = tf.expand_dims(inp_sentence, 0)
	if raw_tokens:
		encoder_input = inp_sentence
	if len(encoder_input.shape) == 1:
		encoder_input = tf.expand_dims(encoder_input, 0)
	assert len(encoder_input.shape) == 2  
	
	decoder_input = [tokenizer_tar.vocab_size]
	output = tf.expand_dims(decoder_input, 0)
	i = len(decoder_input) #i = the length of the currently generated translation
	token_type = tf.concat([
		tf.zeros(tf.shape(output[:, :-1]),dtype=tf.int64),
		tf.ones(tf.shape(output[:, -1:]),dtype=tf.int64)
	], axis=-1)
 
	while i <= max_length + mem_tokens_num:
		i += 1
		enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)    
		predictions, enc_att_w, attention_weights = transformer(
			encoder_input, output, False, enc_padding_mask,
			combined_mask, dec_padding_mask, token_type_inp=token_type)

		token_id_preds = predictions[:, :, :-2]
		predicted_token_type = tf.cast(tf.argmax(predictions[: , -1:, -2:], axis=-1), tf.int64)
		# bound еру number of generated memory tokens 
		if tf.cast(tf.shape(token_type[:, 1:])[1], 
				   dtype=tf.int64) - tf.math.reduce_sum(token_type[:, 1:]) > mem_tokens_num - 1:
			predicted_token_type = tf.constant([[1]], dtype=tf.int64)
		token_type = tf.concat([token_type, predicted_token_type], axis=-1)
		predicted_id = tf.cast(tf.map_fn(sample_tokens, 
										 (token_id_preds[:, -1:, :],
										  predicted_token_type,
										  tf.ones(tf.shape(token_id_preds)[0]) * nucleus_p),
										 fn_output_signature=tf.TensorSpec(shape=(None),
																		   dtype=tf.int64)),
							   tf.int32)
		if predicted_id == tokenizer_tar.vocab_size + 1:
			output = tf.concat([output, predicted_id], axis=-1)
			final_res = tf.squeeze(output, axis=0)
			if raw_tokens:
				predicted_sentence = []
				for j,i in enumerate(final_res):
					if token_type[0, j] == 1:
						predicted_sentence.append(i)
				return predicted_sentence, enc_att_w, attention_weights, token_type
			else:
				return final_res, enc_att_w, attention_weights, token_type

		output = tf.concat([output, predicted_id], axis=-1)
	final_res = tf.squeeze(output, axis=0)

	if raw_tokens:
		predicted_sentence = []
		for j, i in enumerate(final_res):
			if token_type[0, j] == 1:
				predicted_sentence.append(i)
		return predicted_sentence, enc_att_w, attention_weights, token_type
	else:
		return final_res, enc_att_w, attention_weights, token_type
		
def translate(sentence, tokenizer_inp, tokenizer_tar, 
			  max_length, mem_tokens_num, plot=[]):
	result, enc_attn_w, attention_weights, token_type = generate(sentence, 
																 tokenizer_inp, 
																 tokenizer_tar,
																 max_length,
																 mem_tokens_num
																)
	predicted_sequence = []
	predicted_mem = []
	predicted_sentence = []
	for j, i in enumerate(result):
		if i < tokenizer_tar.vocab_size:
			if token_type[0, j] == 1:
				predicted_sentence.append(tokenizer_tar.decode([i]))
				predicted_sequence.append(tokenizer_tar.decode([i]))
			else:
				predicted_mem.append(tokenizer_tar.decode([i]))
				predicted_sequence.append(tokenizer_tar.decode([i]))
		if i == tokenizer_tar.vocab_size:
			if token_type[0, j] == 1:
				predicted_sentence.append('<start>')
				predicted_sequence.append('<start>')
			else:
				predicted_mem.append('<start>')
				predicted_sequence.append('<start>')
		if i == tokenizer_tar.vocab_size + 1:
			if token_type[0, j] == 1:
				predicted_sentence.append('<end>')
				predicted_sequence.append('<end>')
			else:
				predicted_mem.append('<end>')
				predicted_sequence.append('<end>')
		if i == tokenizer_tar.vocab_size + 2:
			if token_type[0, j] == 1:
				predicted_sentence.append('<mem>')
				predicted_sequence.append('<mem>') 
			else:
				predicted_mem.append('<mem>')
				predicted_sequence.append('<mem>')
	if len(plot) == 0:
		print('Input: {}'.format(sentence))
		print('Predicted sequence: {}'.format(''.join(predicted_sequence)))
		print('Predicted memory : ({})'.format(')('.join(predicted_mem)))
		print('Predicted translation: {}'.format(''.join(predicted_sentence)))

def _get_ngrams(segment, max_order):
	"""Extracts all n-grams upto a given maximum order from an input segment.
	Args:
		segment: text segment from which n-grams will be extracted.
		max_order: maximum length in tokens of the n-grams returned by this
				methods.
	Returns:
		The Counter containing all n-grams upto max_order in segment
		with a count of how many times each n-gram occurred.
	"""
	ngram_counts = collections.Counter()
	for order in range(1, max_order + 1):
		for i in range(0, len(segment) - order + 1):
			ngram = tuple(segment[i:i+order])
			ngram_counts[ngram] += 1
	return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
	"""Computes BLEU score of translated segments against one or more references.
	Args:
		reference_corpus: list of lists of references for each translation. Each
				reference should be tokenized into a list of tokens.
		translation_corpus: list of translations to score. Each translation
				should be tokenized into a list of tokens.
		max_order: Maximum n-gram order to use when computing BLEU score.
		smooth: Whether or not to apply Lin et al. 2004 smoothing.
	Returns:
		3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
		precisions and brevity penalty.
	"""
	matches_by_order = [0] * max_order
	possible_matches_by_order = [0] * max_order
	reference_length = 0
	translation_length = 0
	for (references, translation) in zip(reference_corpus,
										 translation_corpus):
		reference_length += min(len(r) for r in references)
		translation_length += len(translation)

		merged_ref_ngram_counts = collections.Counter()
		for reference in references:
			merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
		translation_ngram_counts = _get_ngrams(translation, max_order)
		overlap = translation_ngram_counts & merged_ref_ngram_counts
		for ngram in overlap:
			matches_by_order[len(ngram)-1] += overlap[ngram]
		for order in range(1, max_order+1):
			possible_matches = len(translation) - order + 1
			if possible_matches > 0:
				possible_matches_by_order[order-1] += possible_matches

	precisions = [0] * max_order
	for i in range(0, max_order):
		if smooth:
			precisions[i] = ((matches_by_order[i] + 1.) /
											 (possible_matches_by_order[i] + 1.))
		else:
			if possible_matches_by_order[i] > 0:
				precisions[i] = (float(matches_by_order[i]) /
												 possible_matches_by_order[i])
			else:
				precisions[i] = 0.0

	if min(precisions) > 0:
		p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
		geo_mean = math.exp(p_log_sum)
	else:
		geo_mean = 0

	ratio = float(translation_length) / reference_length

	if ratio > 1.0:
		bp = 1.
	else:
		bp = math.exp(1 - 1. / ratio)

	bleu = geo_mean * bp

	return (bleu, precisions, bp, ratio, translation_length, reference_length)

def explain_bleu(bleu_values):
	bleu, precisions, bp, ratio, translation_length, reference_length = bleu_values

	print(f"BLEU score: {bleu:.4}")
	print("----------------")
	print(f"Translated text total length\t {translation_length}")
	print(f"Reference text total length\t {translation_length}")
	print("----------------")
	print(f"n-gram max order was {len(precisions)}")
	print("n-gram precisions: ", end="")
	for val in precisions:
		print(f"{val:.3}", end=" ")
	print()
	print(f"Brevity penalty: {bp}")

def meteor_detokenize(s, tokenizer_tar):
	predicted_sequence = []
	predicted_mem = []
	predicted_sentence = []
	for j, i in enumerate(s):
		if i < tokenizer_tar.vocab_size:
			predicted_sentence.append(tokenizer_tar.decode([i]))
			predicted_sequence.append(tokenizer_tar.decode([i]))
		if i == tokenizer_tar.vocab_size:
			predicted_sentence.append('<start>')
			predicted_sequence.append('<start>')
		if i == tokenizer_tar.vocab_size + 1:
			predicted_sentence.append('<end>')
			predicted_sequence.append('<end>')
		if i == tokenizer_tar.vocab_size + 2:
			print('mem found')
			predicted_sentence.append('<mem>')
			predicted_sequence.append('<mem>') 
	return ''.join(predicted_sentence), ''.join(predicted_sequence), ''.join(predicted_mem)


def evaluate(config):
	val_preprocessed, tokenizer_inp, tokenizer_tar = load_data(
		data_path=config['data_path'], tokenizer_path=config['tokenizer_path'],
		max_length=config['max_length'], split='validation')
	
	transformer = Transformer(config['num_layers'], config['d_model'],
							  config['num_heads'], config['dff'],
							  input_vocab_size, target_vocab_size,
							  pe_input=input_vocab_size,
							  pe_target=target_vocab_size,
							  rate=config['dropout_rate'],
							  token_type=config['token_type'])
	ckpt = tf.train.Checkpoint(transformer=transformer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, config['checkpoint_path'], max_to_keep=None) 
	ckpt.restore(os.path.join(checkpoint_path, 
							  f"ckpt-{config['epochs_with_mem']}")
				).assert_existing_objects_matched()
	
	refs = [np.array(ref[:-1]) for (_, ref) in tqdm(val_preprocessed)]
	trans = [np.array(generate(inp, tokenizer_inp, tokenizer_tar,
							   config['max_length'], 
							   config['mem_tokens_num'],
							   raw_tokens=True)[0][:-1]) for (inp, _) in tqdm(val_preprocessed)]
	
	# BLEU-4
	explain_bleu(compute_bleu([[r] for r in refs], trans))

	# METEOR
	scores = [
		single_meteor_score(meteor_detokenize(ref, tokenizer_tar)[0], 
							meteor_detokenize(pred, tokenizer_tar)[0], 
							alpha=0.9, beta=3, gamma=0.5)
		for ref, pred in tqdm(zip(refs, trans))
		]
	print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} METEOR = {np.mean(scores)}, {len(refs)} samples\n")

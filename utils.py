import os
import tensorflow as tf
import tensorflow_datasets as tfds


def create_tokenizers(tokenizer_path, examples=None):
	if examples is not None:
		tokenizer_inp = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
			(inp.numpy() for inp, _ in examples), target_vocab_size=2**13)
		tokenizer_tar = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
			(tar.numpy() for _, tar in examples), target_vocab_size=2**13)
		tokenizer_inp.save_to_file(os.path.join(tokenizer_path, 'inputs_tokenizer'))
		tokenizer_tar.save_to_file(os.path.join(tokenizer_path, 'targets_tokenizer'))
	else:
		tokenizer_inp = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
			os.path.join(tokenizer_path, 'inputs_tokenizer')
		)
		tokenizer_tar = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
			os.path.join(tokenizer_path, 'targets_tokenizer')
		)
	return tokenizer_inp, tokenizer_tar


def encode(lang1, lang2):
	lang1 = [tokenizer_inp.vocab_size] + tokenizer_inp.encode(lang1.numpy()) + [tokenizer_inp.vocab_size + 1]
	lang2 = [tokenizer_tar.vocab_size] + tokenizer_tar.encode(lang2.numpy()) + [tokenizer_tar.vocab_size + 1]
	return lang1, lang2

def tf_encode(inp, tar):
	result_inp, result_tar = tf.py_function(encode, [inp, tar], [tf.int64, tf.int64])
	result_inp.set_shape([None])
	result_tar.set_shape([None])
	return result_inp, result_tar

def filter_max_length(x, y):
	return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

def filter_min_length(x, y):
	return tf.logical_and(tf.size(x) >= min_length, tf.size(y) >= min_length)

def load_data(data_path, tokenizer_path, max_length=40, split='train'):
	if data_path == 'it' or data_path == 'opsub':
		return load_it_opsub(data_spec=data_path)
	elif data_path == 'wsc':
		return load_wsc()
	else:
		examples, _ = tfds.load(data_path, with_info=True, as_supervised=True)
		if split is not None:
			examples = examples[split]
		if not os.path.exists(tokenizer_path):
			tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path, examples) 
		else:
			tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path)
	
		dataset = examples.map(lambda x, y: tf_encode(x, y)).filter(filter_max_length)
		return dataset, tokenizer_inp, tokenizer_tar

def load_wsc(split='train'):
	tokenizer_path = './tokenizers/wsc'
	ru_train = tf.data.TextLineDataset(['./wsc-ru-en/ru_train.txt'])
	en_train = tf.data.TextLineDataset(['./wsc-ru-en/en_train.txt'])
	wsc_ru_en_train = tf.data.Dataset.from_generator(lambda: zip(ru_train, en_train),
													 output_types=(tf.string, tf.string),
													 output_shapes=(None, None)
													)
	
	ru_val = tf.data.TextLineDataset(['./wsc-ru-en/ru_val.txt'])
	en_val = tf.data.TextLineDataset(['./wsc-ru-en/en_val.txt'])
	wsc_ru_en_val = tf.data.Dataset.from_generator(lambda: zip(ru_val, en_val), 
												   output_types=(tf.string, tf.string),
												   output_shapes=(None, None)
												  )
	
	ru_test = tf.data.TextLineDataset(['./wsc-ru-en/ru_test.txt'])
	en_test = tf.data.TextLineDataset(['./wsc-ru-en/en_test.txt'])
	wsc_ru_en_test = tf.data.Dataset.from_generator(lambda: zip(ru_test, en_test), 
													output_types=(tf.string, tf.string),
													output_shapes=(None, None)
												   )
	wsc_ru_en_test = wsc_ru_en_test.concatenate(wsc_ru_en_val)
	min_length = 7
	max_length = 140

	if not os.path.exists(tokenizer_path):
		tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path,
														 wsc_ru_en_train.concatenate(
															 wsc_ru_en_test)
														) 
	else:
		tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path)

	tf.data.experimental.save(wsc_ru_en_train, 'wsc_train', compression=None, shard_func=None)
	tf.data.experimental.save(wsc_ru_en_test, 'wsc_test', compression=None, shard_func=None)
	if split == 'train':
		return wsc_ru_en_train, tokenizer_inp, tokenizer_tar
	else:
		return wsc_ru_en_test, tokenizer_inp, tokenizer_tar

def load_it_opsub(data_spec='opsub'):
	if data_spec == 'it':
		config = tfds.translate.opus.OpusConfig(
			version=tfds.core.Version('0.1.0'),
			language_pair=("ru", "en"),
			subsets=["GNOME", "KDE4", "PHP", "Ubuntu", "OpenOffice"]
		)
	elif data_spec == 'opsub':
		config = tfds.translate.opus.OpusConfig(
			version=tfds.core.Version('0.1.0'),
			language_pair=("ru", "en"),
			subsets=["OpenSubtitles"]
		)
	examples = tfds.builder("opus", config=config).download_and_prepare().as_dataset(
		split='train',
		shuffle_files=True
	)
	min_length = 7
	max_length = 140
	tokenizer_path = f'./tokenizers/{data_spec}'
	if not os.path.exists(tokenizer_path):
		tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path, it_examples) 
	else:
		tokenizer_inp, tokenizer_tar = create_tokenizers(tokenizer_path)

	dataset = examples.map(lambda x: tf_encode(x['ru'], x['en'])).filter(
		filter_min_length).filter(filter_max_length)
	
	return dataset, tokenizer_inp, tokenizer_tar

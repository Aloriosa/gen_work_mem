from train import train
from evaluate import evaluate


def main():
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
		"data_path": 'ted_hrlr_translate/ru_to_en',
		"tokenizer_path": './tokenizers/ru-en',
		"max_length": 140,
		"buffer_size": 20000,
		"batch_size": 64,
		"mem_tokens_num": 10,
		"nucleus_p": 0.9,
		"checkpoint_path": './ckpts_work_mem_nucleus_0.9',
		"epochs_pretrain": 5,
		"epochs_with_mem": 20,
		}
	train(config)
	evaluate(config)
	# tuning on opsub subtask. Instead of opsub could be wsc/ted/it
	config.update(
		{
			"data_path": 'opsub',
			"tokenizer_path": './tokenizers/opsub',
			"epochs_pretrain": 20,
			"epochs_with_mem": 50,
		}
	)
	train(config)

if __name__ == "__main__":
	main()

export PYTHONHASHSEED=0
export output_dir="./seed0_rank8/000110001111/rte"

python3 finetune_encoder.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 32 \
--learning_rate 1.2e-3 \
--num_train_epochs 50 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--eval_strategy steps \
--eval_steps 300 \
--save_strategy steps \
--save_steps 3000 \
--warmup_ratio 0.1 \
--lora_r 8 \
--lora_dropout 0.0 \
--lora_layers 000110001111 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.01 \
--fp16 True 

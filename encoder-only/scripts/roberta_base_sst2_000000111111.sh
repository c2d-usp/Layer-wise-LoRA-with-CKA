export PYTHONHASHSEED=0
export output_dir="./seed0_rank8/000000111111/sst2"

python3 finetune_encoder.py \
--model_name_or_path roberta-base \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 16 \
--learning_rate 5e-4 \
--num_train_epochs 60 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--lora_r 8 \
--lora_dropout 0.1 \
--lora_layers 000000111111 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1 \
--fp16 True \
--lr_scheduler_type linear \
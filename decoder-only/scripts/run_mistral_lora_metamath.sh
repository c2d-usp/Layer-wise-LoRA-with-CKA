BASE_MODEL="mistralai/Mistral-7B-v0.1"
OUTPUT_PATH="output/metamath-LoRA-mistral-7b-r128-11111111111110100010000000000001"
DATA_PATH="fxmeng/pissa-dataset"

python3 finetune_decoder.py \
    --model_name_or_path $BASE_MODEL \
    --bf16 \
    --init_weights standard \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --lora_layers 11111111111110100010000000000001 \
    --data_path $DATA_PATH \
    --sub_task metamath:100000 \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --seed 0 \
    --lr_scheduler_type "cosine" \
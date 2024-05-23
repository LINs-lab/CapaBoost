export http_proxy=http://192.168.123.169:18889
export https_proxy=http://192.168.123.169:18889
export HTTP_PROXY=http://192.168.123.169:18889
export HTTPS_PROXY=http://192.168.123.169:18889

export CUDA_VISIBLE_DEVICES=0

num_layer=2
density=0.5
seed=42

# lora
adapter_config=lora
lora_r=1
lora_alpha=16

# optimization
learning_rate=1e-3
epoch=15

TASK_NAME=squad
exp_name=multi-mask.weight-tied.lora.sd_${seed}.lora_r_${lora_r}.lora_alpha_${lora_alpha}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${epoch}.specifc_epoch
SAVE=checkpoints/${TASK_NAME}/parallel-weight-tied_lora/${exp_name}/

python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --do_train \
    --do_eval \
    --dataset_name ${TASK_NAME} \
    --learning_rate ${learning_rate} \
    --output_dir ${SAVE} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --warmup_steps 1000 \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --weight_decay 0.01 \
    --save_total_limit 2 \
    --train_adapter \
    --adapter_config ${adapter_config} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --num_layer ${num_layer} \
    --density ${density} \
    --seed ${seed} \
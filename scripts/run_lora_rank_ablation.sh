export http_proxy=http://192.168.123.169:18889
export https_proxy=http://192.168.123.169:18889
export HTTP_PROXY=http://192.168.123.169:18889
export HTTPS_PROXY=http://192.168.123.169:18889

export CUDA_VISIBLE_DEVICES=0
# export TASK_NAME=mnli

# export WANDB_API_KEY=None


# define metric
declare -A metrics
metrics['cola']=matthews_correlation
metrics['sst2']=accuracy
metrics['mrpc']=combined_score
metrics['qqp']=combined_score
metrics['stsb']=combined_score
metrics['rte']=accuracy
metrics['qnli']=accuracy
metrics['mnli']=accuracy

declare -A t2epoch
t2epoch['cola']=40
t2epoch['sst2']=20
t2epoch['mrpc']=40
t2epoch['qqp']=20
t2epoch['stsb']=20
t2epoch['rte']=30
t2epoch['qnli']=10
t2epoch['mnli']=20

declare -A t2lr
t2lr['cola']=5e-4
t2lr['sst2']=1e-4
t2lr['mrpc']=5e-4
t2lr['qqp']=5e-4
t2lr['stsb']=5e-4
t2lr['rte']=5e-4
t2lr['qnli']=5e-4
t2lr['mnli']=5e-4

num_layer=2
density=0.5
sparsity=0.5
mask_lr=3e-2
# learning_rate=1e-4

extra_cmd_for_prefix=""

# small 'rte' 'mrpc' 'stsb' 'cola'
# large 'sst2' 'qqp' 'qnli' 'mnli'

# lora
adapter_config=lora
lora_r=4
lora_alpha=6
# lora_alpha=32


# for TASK_NAME in 'sst2' 'qqp' 'qnli' 'mnli' 'rte' 'mrpc' 'stsb' 'cola'
for TASK_NAME in 'mnli'
do
  export WANDB_PROJECT=Weight-tied-lora.${TASK_NAME}
  # export WANDB_PROJECT=test

  metric=${metrics[${TASK_NAME}]}

  extra_cmd=""

  for seed in 44
  do
    # share
    exp_name=multi-mask_weight-tied.sd_${seed}.lora_r_${lora_r}.lora_alpha_${lora_alpha}.layer_num_${num_layer}.density_${density}.lr_${t2lr[${TASK_NAME}]}.epoch_${t2epoch[${TASK_NAME}]}.specifc_epoch
    extra_cmd="--share_adapter"

    # not share
    # exp_name=original.sd_${seed}.lora_r_${lora_r}.lora_alpha_${lora_alpha}.specifc_epoch

    SAVE=checkpoints/${TASK_NAME}/weight-tied/${exp_name}/
    SAVE_FILE=checkpoints/${TASK_NAME}/weight-tied/${exp_name}/test_results.json

    rm -rf ${SAVE}; mkdir -p ${SAVE}

    until [ -e ${SAVE_FILE} ]
    do
      python examples/pytorch/text-classification/run_glue.py \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --fp16 \
        --max_seq_length 128 \
        --per_device_train_batch_size 128 \
        --learning_rate ${t2lr[${TASK_NAME}]} \
        --mask_learning_rate ${mask_lr} \
        --num_train_epochs ${t2epoch[${TASK_NAME}]} \
        --lora_r ${lora_r} \
        --lora_alpha ${lora_alpha} \
        --num_layer ${num_layer} \
        --density ${density} \
        --overwrite_output_dir \
        --train_adapter \
        --warmup_ratio 0.1 \
        --save_total_limit=2 \
        --adapter_config ${adapter_config} \
        --evaluation_strategy epoch \
        --sparsity ${sparsity} \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --metric_for_best_model ${metric} \
        --weight_decay 0.1 \
        --run_name ${TASK_NAME}.${exp_name} \
        --output_dir ${SAVE} \
        --seed ${seed} #${extra_cmd}
      done
  done
done
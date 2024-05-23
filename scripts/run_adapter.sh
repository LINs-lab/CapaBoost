# export http_proxy=http://192.168.123.169:18889
# export https_proxy=http://192.168.123.169:18889
# export HTTP_PROXY=http://192.168.123.169:18889
# export HTTPS_PROXY=http://192.168.123.169:18889

# export CUDA_VISIBLE_DEVICES=0
# # export TASK_NAME=mnli

# # export WANDB_API_KEY=None


# # define metric
# declare -A metrics
# metrics['cola']=matthews_correlation
# metrics['sst2']=accuracy
# metrics['mrpc']=combined_score
# metrics['qqp']=combined_score
# metrics['stsb']=combined_score
# metrics['rte']=accuracy
# metrics['qnli']=accuracy
# metrics['mnli']=accuracy

# declare -A t2epoch
# t2epoch['cola']=20
# t2epoch['sst2']=10
# t2epoch['mrpc']=20
# t2epoch['qqp']=10
# t2epoch['stsb']=20
# t2epoch['rte']=20
# t2epoch['qnli']=10
# t2epoch['mnli']=10

# num_layer=2
# # density=0.5
# sparsity=0.5
# mask_lr=3e-3
# learning_rate=5e-4

# extra_cmd_for_prefix=""

# # small 'rte' 'mrpc' 'stsb' 'cola'
# # large 'sst2' 'qqp' 'qnli' 'mnli'

# # adapter
# adapter_config=pfeiffer
# adapter_reduction_factor=12


# # for TASK_NAME in 'rte' 'mrpc' 'stsb' 'cola' 'sst2' 'qqp' 'qnli' 'mnli'
# for TASK_NAME in 'rte'
# do
#   export WANDB_PROJECT=ProPETadapter.${TASK_NAME}

#   metric=${metrics[${TASK_NAME}]}

  

#   extra_cmd=""
#   for density in 0.5 0.7 0.9
#   do
#     for seed in 42 43 44
#     # for seed in 44
#     do
#       # share
#       exp_name=multi-mask.weight-tied.adapter.sd_${seed}.arf_${adapter_reduction_factor}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${t2epoch[${TASK_NAME}]}.specifc_epoch
#       extra_cmd="--share_adapter"

#       # not share
#       # exp_name=original.sd_${seed}.arf_${adapter_reduction_factor}.specifc_epoch

#       SAVE=checkpoints/${TASK_NAME}/weight-tied_adapter/${exp_name}/
#       SAVE_FILE=checkpoints/${TASK_NAME}/weight-tied_adapter/${exp_name}/test_results.json

#       rm -rf ${SAVE}; mkdir -p ${SAVE}


#       until [ -e ${SAVE_FILE} ]
#       do
#         python examples/pytorch/text-classification/run_glue.py \
#           --model_name_or_path roberta-base \
#           --task_name $TASK_NAME \
#           --do_train \
#           --do_eval \
#           --do_predict \
#           --fp16 \
#           --max_seq_length 128 \
#           --per_device_train_batch_size 128 \
#           --learning_rate ${learning_rate} \
#           --mask_learning_rate ${mask_lr} \
#           --num_train_epochs ${t2epoch[${TASK_NAME}]} \
#           --num_layer ${num_layer} \
#           --density ${density} \
#           --overwrite_output_dir \
#           --train_adapter \
#           --warmup_ratio 0.1 \
#           --save_total_limit=2 \
#           --adapter_config ${adapter_config} \
#           --evaluation_strategy epoch \
#           --sparsity ${sparsity} \
#           --save_strategy epoch \
#           --load_best_model_at_end True \
#           --metric_for_best_model ${metric} \
#           --weight_decay 0.1 \
#           --run_name ${TASK_NAME}.${exp_name} \
#           --output_dir ${SAVE} \
#           --seed ${seed} \
#           --adapter_reduction_factor ${adapter_reduction_factor} #${extra_cmd} \

#       done
#     done
#   done
# done

export http_proxy=http://192.168.123.169:18889
export https_proxy=http://192.168.123.169:18889
export HTTP_PROXY=http://192.168.123.169:18889
export HTTPS_PROXY=http://192.168.123.169:18889

export CUDA_VISIBLE_DEVICES=5
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
t2epoch['cola']=30
t2epoch['sst2']=10
t2epoch['mrpc']=30
t2epoch['qqp']=10
t2epoch['stsb']=30
t2epoch['rte']=30
t2epoch['qnli']=10
t2epoch['mnli']=10

# num_layer=1
density=0.5
sparsity=0.5
mask_lr=3e-3
# learning_rate=5e-4

extra_cmd_for_prefix=""

# small 'rte' 'mrpc' 'stsb' 'cola'
# large 'sst2' 'qqp' 'qnli' 'mnli'

# adapter
adapter_config=pfeiffer
adapter_reduction_factor=24


# for TASK_NAME in 'rte' 'mrpc' 'stsb' 'cola' 'sst2' 'qqp' 'qnli' 'mnli'
for TASK_NAME in 'mnli'
do
  export WANDB_PROJECT=ProPETadapter.${TASK_NAME}

  metric=${metrics[${TASK_NAME}]}

  

  extra_cmd=""
  for num_layer in 2
  do
    for epoch in 30
    do
      # for learning_rate in 1e-3 3e-3 5e-3 1e-4 8e-4
      for learning_rate in 5e-4
      do
        for seed in 42
        # for seed in 44
        do
          # share
          # exp_name=multi-mask.weight-tied.adapter.sd_${seed}.arf_${adapter_reduction_factor}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${t2epoch[${TASK_NAME}]}.specifc_epoch
          exp_name=multi-mask.weight-tied.adapter.sd_${seed}.arf_${adapter_reduction_factor}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${epoch}.larger_bs
          extra_cmd="--share_adapter"

          # not share
          # exp_name=original.sd_${seed}.arf_${adapter_reduction_factor}.specifc_epoch

          SAVE=checkpoints/${TASK_NAME}/deberta/parallel-weight-tied_adapter/${exp_name}/
          SAVE_FILE=checkpoints/${TASK_NAME}/deberta/parallel-weight-tied_adapter/${exp_name}/test_results.json

          rm -rf ${SAVE}; mkdir -p ${SAVE}


          until [ -e ${SAVE_FILE} ]
          do
            python examples/pytorch/text-classification/run_glue.py \
              --model_name_or_path microsoft/deberta-v3-base \
              --task_name $TASK_NAME \
              --do_train \
              --do_eval \
              --do_predict \
              --fp16 \
              --max_seq_length 256 \
              --per_device_train_batch_size 64 \
              --learning_rate ${learning_rate} \
              --mask_learning_rate ${mask_lr} \
              --num_train_epochs ${epoch} \
              --num_layer ${num_layer} \
              --density ${density} \
              --overwrite_output_dir False \
              --train_adapter \
              --warmup_steps 1000 \
              --save_total_limit=2 \
              --adapter_config ${adapter_config} \
              --evaluation_strategy epoch \
              --sparsity ${sparsity} \
              --save_strategy epoch \
              --load_best_model_at_end True \
              --metric_for_best_model ${metric} \
              --weight_decay 0.01 \
              --run_name ${TASK_NAME}.${exp_name} \
              --output_dir ${SAVE} \
              --seed ${seed} \
              --adapter_reduction_factor ${adapter_reduction_factor} #${extra_cmd} \

          done
        done
      done
    done
  done
done
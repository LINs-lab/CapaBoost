# export http_proxy=http://192.168.123.169:18889
# export https_proxy=http://192.168.123.169:18889
# export HTTP_PROXY=http://192.168.123.169:18889
# export HTTPS_PROXY=http://192.168.123.169:18889

export CUDA_VISIBLE_DEVICES=0

sparsity=0.5
mask_lr=3e-2
num_layer=2
density=0.5
seed=42

# adapter
adapter_config=pfeiffer
adapter_reduction_factor=12

# optimization
learning_rate=5e-4
epoch=25

TASK_NAME=cnn_dailymail
exp_name=multi-mask.weight-tied.adapter.sd_${seed}.arf_${adapter_reduction_factor}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${epoch}.specifc_epoch
SAVE=checkpoints/${TASK_NAME}/weight-tied_adapter/${exp_name}/

python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name ${TASK_NAME} \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --learning_rate ${learning_rate} \
    --output_dir ${SAVE} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=16 \
    --max_source_length 512 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --warmup_steps 500 \
    --num_beams 8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --load_best_model_at_end \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --max_steps 100000 \
    --evaluation_strategy epoch \
    --weight_decay 0.01 \
    --save_total_limit 2 \
    --save_strategy epoch \
    --train_adapter \
    --adapter_config ${adapter_config} \
    --num_layer ${num_layer} \
    --density ${density} \
    --adapter_reduction_factor ${adapter_reduction_factor} \
    --seed ${seed} \
export http_proxy=http://192.168.123.169:18889
export https_proxy=http://192.168.123.169:18889
export HTTP_PROXY=http://192.168.123.169:18889
export HTTPS_PROXY=http://192.168.123.169:18889

export CUDA_VISIBLE_DEVICES=0

num_layer=2
density=0.5
seed=42

# adapter
adapter_config=pfeiffer
adapter_reduction_factor=24

# optimization
learning_rate=1e-3
epoch=20

TASK_NAME=squad
exp_name=multi-mask.weight-tied.adapter.sd_${seed}.arf_${adapter_reduction_factor}.num_layer_${num_layer}.density_${density}.lr_${learning_rate}.num_epoch_${epoch}.specifc_epoch-test
SAVE=checkpoints/${TASK_NAME}/parallel-weight-tied_adapter/${exp_name}/

python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --do_train \
    --do_eval \
    --dataset_name ${TASK_NAME} \
    --learning_rate ${learning_rate} \
    --output_dir ${SAVE} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 16 \
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
    --adapter_reduction_factor ${adapter_reduction_factor} \
    --num_layer ${num_layer} \
    --density ${density} \
    --seed ${seed} \
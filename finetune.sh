# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
gpu_num=$(nvidia-smi --list-gpus | wc -l)

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
model_name_or_model_dir="iic/SenseVoiceSmall"

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
train_data=${workspace}/cantonese/train.jsonl
val_data=${workspace}/cantonese/dev.jsonl

# exp output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool=/notebooks/FunASR/funasr/bin/train_ds.py

torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=2 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=1000 \
++train_conf.save_checkpoint_interval=1000 \
++train_conf.keep_nbest_models=10 \
++train_conf.avg_nbest_model=5 \
++train_conf.use_deepspeed=false \
++train_conf.deepspeed_config=${deepspeed_config} \
++train_conf.use_bf16=true \
++train_conf.grad_clip=1.0 \
++optim=adamw \
++optim_conf.lr=3e-5 \
++optim_conf.weight_decay=0.01
++train_conf.use_wandb=true \
++train_conf.wandb_project="cantonese_asr" \
++train_conf.wandb_team="ming030890" \
++train_conf.wandb_exp_name="funasr-cantonese-$(date +%Y%m%d_%H%M%S)" \
++output_dir="${output_dir}" &> ${log_file}

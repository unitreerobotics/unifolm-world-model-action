# NCCL configuration
# export NCCL_DEBUG=debug
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export CUDA_LAUNCH_BLOCKING=1

# export NCCL_TOPO_FILE=/tmp/topo.txt
# export MASTER_ADDR="master.ip."
# export MASTER_PROT=12366


# args
name="experiment_name"
config_file=configs/train/config.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/path/to/savedir"

mkdir -p $save_root/$name

## run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 --nnodes=1 --master_addr=127.0.0.1 --master_port=12366 --node_rank=0 \
./scripts/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices 8 \
--total_gpus=8 \
lightning.trainer.num_nodes=1

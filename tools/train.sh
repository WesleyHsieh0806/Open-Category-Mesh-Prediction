NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0
WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --use_env \
    tools/train.py
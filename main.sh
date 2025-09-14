set -e
set -x


MODEL=vit_b         # vit_b, vit_l, vit_h

NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12345 --nproc_per_node=8 main.py --checkpoint pretrained_checkpoint/$MODEL.pth --model-type $MODEL --output work_dirs/spt+$MODEL --eval --restore-model spt_ckpt/spt_$MODEL.pth



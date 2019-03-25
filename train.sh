#!/bin/sh

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.141.8.84" --master_port=9876 tools/train_net.py --config-file configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2_lsvt.yaml

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=1 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=2 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=3 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/aw6g5gu5 &
sleep 4s
CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/aw6g5gu5

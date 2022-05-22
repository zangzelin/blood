#!/bin/bash

CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/42wc3tvk &
sleep 4s
CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/42wc3tvk &

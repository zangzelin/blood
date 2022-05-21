#!/bin/bash
# python baseline.py --knn_neighbors 2 &
# python baseline.py --knn_neighbors 3 &
# python baseline.py --knn_neighbors 4 &
# python baseline.py --knn_neighbors 5 &
# python baseline.py --knn_neighbors 6 &
# python baseline.py --knn_neighbors 7 &
# python baseline.py --knn_neighbors 8 &
# python baseline.py --knn_neighbors 9 &
# python baseline.py --knn_neighbors 10 &
# python baseline.py --knn_neighbors 11 &
# python baseline.py --knn_neighbors 12 &
CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=1 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=2 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=3 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
sleep 4s
CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/66f6xcoy &

# sleep 4s
# CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=1 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=2 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=3 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s
# CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# sleep 4s

# CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=1 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=2 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=3 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/66f6xcoy &

# CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=1 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=2 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=3 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=4 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=5 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=6 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=7 wandb agent cairi/bloodcenter_zzl/66f6xcoy &

# CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy &
# CUDA_VISIBLE_DEVICES=0 wandb agent cairi/bloodcenter_zzl/66f6xcoy

# echo "Device ID:" $1
# CUDA_VISIBLE_DEVICES=$1 python main.py --foldindex $[ $1*5+0 ] &&
# CUDA_VISIBLE_DEVICES=$1 python main.py --foldindex $[ $1*5+1 ] &&
# CUDA_VISIBLE_DEVICES=$1 python main.py --foldindex $[ $1*5+2 ] &&
# CUDA_VISIBLE_DEVICES=$1 python main.py --foldindex $[ $1*5+3 ] &&
# CUDA_VISIBLE_DEVICES=$1 python main.py --foldindex $[ $1*5+4 ] 
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 5 &
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 6 &
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 7 &
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 8 &
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 9 &

# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 0
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 1
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 2
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 3
# CUDA_VISIBLE_DEVICES=0 python main.py --foldindex 4
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 5
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 6
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 7
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 8
# CUDA_VISIBLE_DEVICES=1 python main.py --foldindex 9

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

CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 0 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 1 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 2 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 3 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 4 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 5 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 6 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 7 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 8 &&
CUDA_VISIBLE_DEVICES=1 python main_0209.py --foldindex 9

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
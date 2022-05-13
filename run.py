import subprocess
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--scale', type=int, default=30)
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--K', type=int, default=15)

    args = parser.parse_args()
    for i in range(10):
        text = f'CUDA_VISIBLE_DEVICE=1 python main.py --K={args.K} --scale={args.scale} --vs={args.vs} --foldindex={i}'
        process = subprocess.Popen(text, shell=True)
    process.wait()

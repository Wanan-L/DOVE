#!/usr/bin/env bash  
# python sample_videos_to_s2.py  
  
import random  
from pathlib import Path  
  
def sample_videos(video_list_path: str, output_path: str, num_samples: int = 200, seed: int = 42):  
    """  
    从视频列表文件中随机采样指定数量的视频  
      
    Args:  
        video_list_path: 原始视频列表文件路径  
        output_path: 采样后的输出文件路径  
        num_samples: 采样数量  
        seed: 随机种子  
    """  
    # 设置随机种子确保可重现  
    random.seed(seed)  
      
    # 读取原始视频列表  
    with open(video_list_path, 'r') as f:  
        video_paths = [line.strip() for line in f.readlines()]  
      
    # 随机采样  
    sampled_videos = random.sample(video_paths, min(num_samples, len(video_paths)))  
      
    # 保存采样结果  
    with open(output_path, 'w') as f:  
        for video_path in sampled_videos:  
            f.write(video_path + '\n')  
      
    print(f"从 {len(video_paths)} 个视频中采样了 {len(sampled_videos)} 个，保存到 {output_path}")  
  
if __name__ == "__main__":  
    # 使用示例  
    sample_videos(  
        video_list_path="../../datasets/train/HQ-VSR.txt",  
        output_path="../../datasets/train/HQ-VSR-800.txt",  
        num_samples=800,  
        seed=42  
    )

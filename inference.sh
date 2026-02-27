#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

# UDM10
# 内置指标计算功能，可一次性完成视频生成推理+质量评估
python finetune/scripts/prepare_dataset.py --dir /data2/wujialing/code/VSR/DOVE/datasets/test/UDM10/GT-Video
python finetune/scripts/prepare_dataset.py --dir /data2/wujialing/code/VSR/DOVE/datasets/test/UDM10/LQ-Video

CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/UDM10/Video \
    --is_vae_st > logs/DOVE_UDM10_log.txt 2>&1 & tail -f logs/DOVE_UDM10_log.txt

# 时间分块
CUDA_VISIBLE_DEVICES=1 nohup python inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/UDM10_chunk16_overlap8/Video \
    --is_vae_st \
    --chunk_len 16 \
    --overlap_t 8 > logs/DOVE_UDM10_chunk16_overlap8_log.txt 2>&1 & tail -f logs/DOVE_UDM10_chunk16_overlap8_log.txt

# 空间分块
CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/UDM10_tile192_overlap16/Video \
    --is_vae_st \
    --tile_size_hw 192 192 \
    --overlap_hw 16 16 > logs/DOVE_UDM10_tile192_overlap16_log.txt 2>&1 & tail -f logs/DOVE_UDM10_tile192_overlap16_log.txt

# --chunk_len 16 \
# --overlap_t 8   \

# 单独计算指标
CUDA_VISIBLE_DEVICES=4 python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results_5B-s2/UDM10 \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

# 自训练权重
CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/UDM10 \
    --is_vae_st > logs_5B-s2/DOVE_UDM10_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_UDM10_log.txt

# ------------------------------------------------------------------------------------------------
# SPMCS
python finetune/scripts/prepare_dataset.py --dir /data2/wujialing/code/VSR/DOVE/datasets/test/SPMCS/GT-Video
python finetune/scripts/prepare_dataset.py --dir /data2/wujialing/code/VSR/DOVE/datasets/test/SPMCS/LQ-Video

CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/SPMCS/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/SPMCS/Video \
    --is_vae_st > logs/DOVE_SPMCS_log.txt 2>&1 & tail -f logs/DOVE_SPMCS_log.txt

CUDA_VISIBLE_DEVICES=7 python inference_script.py \
    --input_dir datasets/test/SPMCS/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/SPMCS \
    --is_vae_st \
    --tile_size_hw 192 192 \
    --overlap_hw 16 16 

CUDA_VISIBLE_DEVICES=4 python eval_metrics.py \
    --gt datasets/test/SPMCS/GT \
    --pred results_5B-s2/SPMCS \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

# 自训练权重
CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/SPMCS/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/SPMCS \
    --is_vae_st > logs_5B-s2/DOVE_SPMCS_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_SPMCS_log.txt

# ------------------------------------------------------------------------------------------------
# YouHQ40
CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/YouHQ40/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/YouHQ40/Video \
    --is_vae_st > logs/DOVE_YouHQ40_log.txt 2>&1 & tail -f logs/DOVE_YouHQ40_log.txt

CUDA_VISIBLE_DEVICES=1 python inference_script.py \
    --input_dir datasets/test/YouHQ40/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/YouHQ40 \
    --is_vae_st \
    --tile_size_hw 192 192 \
    --overlap_hw 16 16

CUDA_VISIBLE_DEVICES=4 python eval_metrics.py \
    --gt datasets/test/YouHQ40/GT \
    --pred results_5B-s2/YouHQ40 \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

# 自训练权重
CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/YouHQ40/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/YouHQ40 \
    --is_vae_st > logs_5B-s2/DOVE_YouHQ40_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_YouHQ40_log.txt    

# ------------------------------------------------------------------------------------------------
# REDS
CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/REDS/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/REDS/Video \
    --is_vae_st > logs/DOVE_REDS_log.txt 2>&1 & tail -f logs/DOVE_REDS_log.txt

CUDA_VISIBLE_DEVICES=4 python eval_metrics.py \
    --gt datasets/test/REDS/GT \
    --pred results_5B-s2/REDS \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/REDS/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/REDS \
    --is_vae_st > logs_5B-s2/DOVE_REDS_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_REDS_log.txt

# ------------------------------------------------------------------------------------------------
# RealVSR
CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/RealVSR/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/RealVSR/Video \
    --is_vae_st \
    --upscale 1 > logs/DOVE_RealVSR_log.txt 2>&1 & tail -f logs/DOVE_RealVSR_log.txt

CUDA_VISIBLE_DEVICES=5 python inference_script.py \
    --input_dir datasets/test/RealVSR/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/RealVSR \
    --is_vae_st \
    --upscale 1 \
    --tile_size_hw 192 192 \
    --overlap_hw 16 16 \
    --chunk_len 16 \
    --overlap_t 8

CUDA_VISIBLE_DEVICES=4 python eval_metrics.py \
    --gt datasets/test/RealVSR/GT \
    --pred results_5B-s2/RealVSR \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

# 自训练权重
CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/RealVSR/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/RealVSR \
    --is_vae_st  \
    --upscale 1 > logs_5B-s2/DOVE_RealVSR_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_RealVSR_log.txt

# ------------------------------------------------------------------------------------------------
# MVSR4x
CUDA_VISIBLE_DEVICES=2 nohup python inference_script.py \
    --input_dir datasets/test/MVSR4x/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/MVSR4x/Video \
    --is_vae_st \
    --upscale 1 > logs/DOVE_MVSR4x_log.txt 2>&1 & tail -f logs/DOVE_MVSR4x_log.txt

python inference_script.py \
    --input_dir datasets/test/MVSR4x/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/MVSR4x \
    --is_vae_st \
    --upscale 1 \

CUDA_VISIBLE_DEVICES=5 python eval_metrics.py \
    --gt datasets/test/MVSR4x/GT \
    --pred results_5B-s2/MVSR4x \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,ilniqe

# 自训练权重
CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/MVSR4x/LQ-Video \
    --model_path finetune/checkpoint/DOVE-s2/ckpt-500-sft\
    --output_path results_5B-s2/MVSR4x \
    --is_vae_st  \
    --upscale 1 > logs_5B-s2/DOVE_MVSR4x_log.txt 2>&1 & tail -f logs_5B-s2/DOVE_MVSR4x_log.txt

# ------------------------------------------------------------------------------------------------
# VideoLQ
CUDA_VISIBLE_DEVICES=1 nohup python inference_script.py \
    --input_dir datasets/test/VideoLQ/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/VideoLQ/Video \
    --is_vae_st \
    --upscale 1 > logs/DOVE_VideoLQ_log.txt 2>&1 & tail -f logs/DOVE_VideoLQ_log.txt

CUDA_VISIBLE_DEVICES=4 python inference_script.py \
    --input_dir datasets/test/VideoLQ/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/VideoLQ \
    --is_vae_st \
    --tile_size_hw 192 192 \
    --overlap_hw 16 16 

CUDA_VISIBLE_DEVICES=3 python eval_metrics.py \
    --pred results_2b-s1/VideoLQ \
    --metrics clipiqa,musiq,niqe,ilniqe


CUDA_VISIBLE_DEVICES=4 nohup python inference_script.py \
    --input_dir datasets/test/VideoLQ/LQ-Video \
    --model_path finetune/checkpoint-2b/DOVE-s1/ckpt-10000-sft \
    --output_path results_2b-s1/VideoLQ \
    --is_vae_st > logs_2b-s1/DOVE_VideoLQ_log.txt 2>&1 & tail -f logs_2b-s1/DOVE_VideoLQ_log.txt

# DOVE-Wan 推理指南

## 概述

`inference_script_wan.py` 是专为 Wan2.1-T2V-1.3B backbone 设计的视频超分辨率推理脚本，支持：

- 单视频/批量视频处理
- 文本提示条件（可选）
- 时间和空间分块（处理长视频或大分辨率）
- 多种评估指标
- PNG 序列或视频输出

## 快速开始

### 1. 基础推理

```bash
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/DiffSynth-Studio"

python inference_script_wan.py \
    --input_dir ./test_videos \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --upscale 4 \
    --dtype bfloat16
```

### 2. 使用训练好的模型

**Stage 1 模型（SFT checkpoint）：**
```bash
# 先转换 checkpoint
python finetune/scripts/prepare_sft_ckpt_wan.py \
    --checkpoint_dir checkpoint/DOVE-Wan-s1/checkpoint-10000

# 推理
python inference_script_wan.py \
    --model_path checkpoint/DOVE-Wan-s1/ckpt-10000-sft \
    --input_dir ./test_videos \
    --output_path ./results_s1
```

**Stage 2 模型（LoRA checkpoint）：**
```bash
python inference_script_wan.py \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --input_dir ./test_videos \
    --output_path ./results_s2
```

### 3. 带文本提示的推理

创建 `prompts.json`：
```json
{
    "video1.mp4": "A high-quality video of a cat playing",
    "video2.mp4": "Clear footage of city traffic"
}
```

运行：
```bash
python inference_script_wan.py \
    --input_dir ./test_videos \
    --input_json prompts.json \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input_dir` | 输入视频目录 | `./test_videos` |
| `--model_path` | 模型路径（SFT checkpoint 或预训练路径） | `checkpoint/DOVE-Wan-s2/ckpt-final` |

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_path` | `./results` | 输出目录 |
| `--upscale` | `4` | 超分倍数 |
| `--upscale_mode` | `bilinear` | 上采样模式 |
| `--dtype` | `bfloat16` | 计算精度（float16/bfloat16/float32） |
| `--fps` | `16` | 输出视频帧率 |
| `--seed` | `42` | 随机种子 |

### 推理控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--noise_step` | `0` | 输入噪声级别（0=无噪声） |
| `--sr_noise_step` | `399` | SR 时间步（Flow Matching） |

### 分块参数（节省显存）

**时间分块：**
```bash
--chunk_len 49 \      # 每块 49 帧
--overlap_t 8         # 重叠 8 帧
```

**空间分块：**
```bash
--tile_size_hw 480 832 \  # 每块 480x832
--overlap_hw 32 32        # 重叠 32x32
```

### 评估参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gt_dir` | Ground truth 目录（用于全参考指标） | `./gt_videos` |
| `--eval_metrics` | 评估指标（逗号分隔） | `psnr,ssim,lpips,dists` |

支持的指标：
- **全参考**：`psnr`, `ssim`, `lpips`, `dists`
- **无参考**：`clipiqa`, `musiq`, `maniqa`, `niqe`

### 输出格式

| 参数 | 说明 |
|------|------|
| `--png_save` | 保存为 PNG 序列（而非视频） |
| `--save_format` | 视频格式：`yuv444p`（无损）或 `yuv420p`（压缩） |

## 使用场景

### 场景 1：处理长视频（节省显存）

```bash
python inference_script_wan.py \
    --input_dir ./long_videos \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --chunk_len 49 \
    --overlap_t 8 \
    --tile_size_hw 480 832 \
    --overlap_hw 32 32
```

### 场景 2：评估模型性能

```bash
python inference_script_wan.py \
    --input_dir ./test_lq \
    --gt_dir ./test_hq \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --eval_metrics psnr,ssim,lpips,dists,clipiqa
```

输出 `results/metrics_psnr_ssim_lpips_dists_clipiqa.json`：
```json
{
  "per_sample": {
    "psnr": [28.5, 29.3, ...],
    "ssim": [0.85, 0.87, ...]
  },
  "average": {
    "psnr": 28.9,
    "ssim": 0.86
  },
  "count": 10
}
```

### 场景 3：保存为 PNG 序列（用于后处理）

```bash
python inference_script_wan.py \
    --input_dir ./test_videos \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --png_save
```

输出结构：
```
results/
├── video1/
│   ├── 000.png
│   ├── 001.png
│   └── ...
└── video2/
    ├── 000.png
    └── ...
```

## 与 CogVideoX 版本的差异

| 特性 | CogVideoX (`inference_script.py`) | Wan (`inference_script_wan.py`) |
|------|-----------------------------------|----------------------------------|
| **Pipeline** | `CogVideoXPipeline` | `WanVideoPipeline` |
| **Scheduler** | `CogVideoXDPMScheduler` | `FlowMatchScheduler` |
| **VAE 输入** | Tensor `[B, C, F, H, W]` | List of `[C, F, H, W]` |
| **Latent 格式** | `[B, F, C, H, W]` (需 permute) | `[B, C, F, H, W]` (无需 permute) |
| **Text Encoder** | T5 (HuggingFace) | UMT5 (DiffSynth) |
| **Timestep dtype** | `torch.long` | `torch.bfloat16` |
| **LoRA 支持** | ✓ (`--lora_path`) | ✗ (LoRA 已融合到 checkpoint) |
| **CPU Offload** | ✓ (`--is_cpu_offload`) | ✗ (Wan 始终在 CUDA) |
| **VAE Slicing/Tiling** | ✓ (`--is_vae_st`) | ✗ (Wan VAE 用 `tiled=False`) |

## 故障排查

### 1. 找不到 DiffSynth-Studio

**错误：**
```
ModuleNotFoundError: No module named 'diffsynth'
```

**解决：**
```bash
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/DiffSynth-Studio"
# 或在脚本中修改 sys.path.insert 的路径
```

### 2. 模型文件缺失

**错误：**
```
FileNotFoundError: diffusion_pytorch_model.safetensors not found
```

**解决：**
- 确保 `model_path` 指向正确的 checkpoint 目录
- 对于 Stage 1，先运行 `prepare_sft_ckpt_wan.py` 转换
- 对于预训练模型，确保 `DIFFSYNTH_MODEL_BASE_PATH` 正确

### 3. 显存不足

**错误：**
```
torch.cuda.OutOfMemoryError
```

**解决：**
```bash
# 启用时间分块
--chunk_len 49 --overlap_t 8

# 启用空间分块
--tile_size_hw 480 832 --overlap_hw 32 32

# 降低精度
--dtype float16
```

### 4. VAE/TextEncoder 缺失

**错误：**
```
Model file not found: Wan2.1_VAE.pth
```

**解决：**
确保 `DIFFSYNTH_MODEL_BASE_PATH` 包含：
```
DiffSynth-Studio/
├── Wan-AI/Wan2.1-T2V-1.3B/
│   ├── diffusion_pytorch_model.safetensors
│   └── google/umt5-xxl/
└── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
    ├── Wan2.1_VAE.safetensors
    └── models_t5_umt5-xxl-enc-bf16.safetensors
```

## 性能优化

### 1. 批处理（单 GPU）

脚本默认逐视频处理（batch_size=1），适合不同长度的视频。

### 2. 多 GPU 推理

当前版本不支持多 GPU。如需多 GPU，可修改：
```python
pipe = WanVideoPipeline.from_pretrained(..., device_map="balanced")
```

### 3. 混合精度

推荐使用 `bfloat16`（Wan 默认训练精度）：
```bash
--dtype bfloat16
```

## 输出示例

**命令：**
```bash
python inference_script_wan.py \
    --input_dir ./test \
    --output_path ./results \
    --model_path checkpoint/DOVE-Wan-s2/ckpt-final \
    --eval_metrics psnr,ssim
```

**输出：**
```
Loading Wan pipeline from checkpoint/DOVE-Wan-s2/ckpt-final
Model loaded
GPU warmup...
Start processing videos
Processing videos: 100%|████████| 5/5 [02:30<00:00, 30.12s/it]

[video1.mp4] Metrics: PSNR=28.45  SSIM=0.8523
[video2.mp4] Metrics: PSNR=29.12  SSIM=0.8634
...

=== Overall Average Metrics ===
PSNR: 28.79
SSIM: 0.8578

All videos processed.
```

**输出文件：**
```
results/
├── video1.mp4
├── video2.mp4
├── ...
└── metrics_psnr_ssim.json
```

## 相关文档

- [Stage 1 分析文档](STAGE1_DOVE_WAN_ANALYSIS.md)
- [prepare_sft_ckpt_wan 分析](PREPARE_SFT_CKPT_WAN_ANALYSIS.md)
- [训练脚本](../finetune/train_ddp_wan_s1.sh)

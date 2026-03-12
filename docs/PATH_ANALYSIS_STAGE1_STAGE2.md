# Stage 1 / Stage 2 路径问题分析

## 一、Stage 1 路径逻辑

### 1.1 配置

| 项目 | 值 |
|------|-----|
| model_path（脚本传入） | `"/data2/wujialing/pretrain/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B"`（绝对路径） |
| load_components 是否使用 model_path | **否**，完全不使用 |

### 1.2 Stage 1 的 load_components 行为

```python
# dove-wan lora_one_s1_trainer.py
model_configs = [
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
]
tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
```

- 仅使用 `model_id="Wan-AI/Wan2.1-T2V-1.3B"`
- DiffSynth 通过 `DIFFSYNTH_MODEL_BASE_PATH` + `model_id` 解析路径
- 实际加载路径：`$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B`

### 1.3 结论：Stage 1 路径无问题

- 不依赖 `model_path` 参数
- 不依赖当前工作目录（cwd）
- 只依赖环境变量 `DIFFSYNTH_MODEL_BASE_PATH`
- 即便传入错误的 `model_path`，Stage 1 也能正常加载

---

## 二、Stage 2 路径逻辑

### 2.1 配置

| 项目 | 值 |
|------|-----|
| model_path（脚本传入） | `"checkpoint/DOVE-Wan-s1/ckpt-10000-sft"`（相对路径） |
| load_components 是否使用 model_path | **是**，用于查找 SFT ckpt 目录 |

### 2.2 Stage 2 的 load_components 行为

```python
model_path = Path(self.args.model_path).resolve()
dit_single = model_path / "diffusion_pytorch_model.safetensors"
# ...
```

- 对 `Path(...).resolve()`：相对路径会按**当前工作目录（cwd）**解析

### 2.3 问题根源

| 运行方式 | cwd | resolve 后的路径 |
|----------|-----|------------------|
| `cd finetune && bash train_ddp_wan_s2.sh` | `.../finetune/` | `.../finetune/checkpoint/DOVE-Wan-s1/ckpt-10000-sft` ✓ |
| `cd DOVE && bash finetune/train_ddp_wan_s2.sh` | `.../DOVE/` | `.../DOVE/checkpoint/...` ✗（ckpt 实际在 finetune 下） |
| `accelerate launch` 从其他目录启动 | 不确定 | 可能错误 ✗ |

即：**Stage 2 使用了相对路径，且 `resolve()` 依赖 cwd，导致路径对运行方式敏感**。

---

## 三、对比总结

| 项目 | Stage 1 | Stage 2 |
|------|---------|---------|
| model_path 传入形式 | 绝对路径 | 相对路径 |
| 是否使用 model_path | 否 | 是 |
| 路径解析方式 | DIFFSYNTH_MODEL_BASE_PATH + model_id | Path(model_path).resolve() |
| 是否依赖 cwd | 否 | 是 |
| 是否存在路径风险 | 否 | 是 |

---

## 四、修改建议

让 Stage 2 的 model_path 解析不依赖 cwd，改为相对训练脚本/项目根目录解析：

**方案 A：以 finetune 目录为基准**

- 获取 `train.py` 所在目录（即 finetune 目录）作为基准
- 当 `model_path` 为相对路径时，按该基准目录解析

**方案 B：在脚本中使用绝对路径**

- 在 `train_ddp_wan_s2.sh` 中把 `model_path` 写成绝对路径
- 例如：`--model_path "$(pwd)/checkpoint/DOVE-Wan-s1/ckpt-10000-sft"`
- 需在 cd 到 finetune 之后再运行脚本

**方案 C：以 output_dir 的父目录为基准**

- 使用 `Path(output_dir).resolve().parent` 作为基准
- 适用于 output_dir 和 model_path 都在同一项目根下（如 finetune）的场景

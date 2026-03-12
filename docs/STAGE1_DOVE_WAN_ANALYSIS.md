# DOVE-Wan 阶段一实现分析与修改指南

> 分析 `lora_one_s1_trainer.py` 实现合理性，给出修改逻辑与 utils.py 影响评估

---

## 一、实现合理性分析

### 1.1 整体架构

`DOVEWanS1Trainer` 将 CogVideoX backbone 替换为 Wan2.1-T2V-1.3B，整体逻辑与 DOVE 两阶段训练一致。下表对比 CogVideoX 与 Wan 的关键差异及当前实现是否正确：

| 组件/步骤 | CogVideoX (dove) | Wan (dove-wan) | 实现正确性 |
|----------|------------------|----------------|------------|
| **Latent 格式** | [B, C, F, H, W] → permute 到 [B, F, C, H, W] | [B, C, F, H, W] 直接传入 | ✅ 正确：`model_fn_wan_video` 内部 `rearrange(x, 'b c f h w -> ...')`，期望 [B, C, F, H, W] |
| **时间步类型** | `torch.long` | `torch.bfloat16` | ✅ 正确：Flow Matching 使用连续时间步 |
| **前向接口** | `transformer(hidden_states, encoder_hidden_states, timestep, image_rotary_emb)` | `model_fn_wan_video(dit, latents, timestep, context, ...)` | ✅ 正确：Wan 使用统一 model_fn |
| **VAE encode** | `vae.encode(video).latent_dist.sample() * scaling_factor` | `vae.encode(video_list, device, tiled=False)` | ✅ 正确：Wan VAE 期望 list of [C,F,H,W] |
| **VAE decode** | `pipe.decode_latents(latent)` | `pipe.vae.decode(latent_list, device, tiled=False)` | ✅ 正确 |
| **损失计算** | `scheduler.get_velocity(pred, x, t)` 后 MSE | 直接 MSE(latent_pred, hq_latent) | ✅ 正确：`model_fn_wan_video` 输出为 velocity，Flow Matching 中预测目标即 velocity |
| **patch_size_t 填充** | 需对 latent 做首帧复制以满足 `patch_size_t` | 不需要 | ✅ 已确认：Wan2.1-T2V-1.3B `patch_size=[1,2,2]`，t=1 对帧数无整除要求 |

### 1.2 正确且合理的部分

1. **`load_components`**  
   - 使用 `WanVideoPipeline.from_pretrained` 加载模型，`fetch_model` 对缺失组件返回 `None`，T2V 只需 dit/vae/text_encoder，逻辑正确。  
   - `components.transformer = pipe.dit` 与 Wan 命名一致。

2. **`encode_video`**  
   - 将 `[B, C, F, H, W]` 转为 `list of [C, F, H, W]`，符合 Wan VAE 接口。  
   - `tiled=False` 适合训练场景。

3. **`encode_text`**  
   - 使用 `(ids, mask) = tokenizer(prompt, return_mask=True)`，与 WanTextEncoder 输入一致。  
   - 按 `seq_lens` 将 padding 置零，符合 Wan 文本编码习惯。

4. **`compute_loss` 与 `model_fn_wan_video`**  
   - 直接调用 `model_fn_wan_video`，latents、timestep、context 格式正确。  
   - 无需 RoPE 外部计算，Wan DiT 内部处理。  
   - 使用 `use_gradient_checkpointing` 参数，符合 Wan 设计。

5. **`prepare_models`**  
   - 不调用父类，避免 `enable_slicing/enable_tiling`（Wan VAE 不支持）。  
   - 用 `SimpleNamespace` 构造 `transformer_config`，供后续逻辑使用。

6. **`prepare_trainable_parameters`**  
   - 对非 `nn.Module` 做 `hasattr` 检查，避免对 tokenizer 等调用 `requires_grad_`。  
   - 不调用 `enable_gradient_checkpointing()`，由 forward 的 `use_gradient_checkpointing` 控制。

### 1.3 需要修正或补充的部分

1. **Wan DiT 的 latent 时间维度对齐**  
   - CogVideoX 有 `patch_size_t` 首帧复制逻辑。  
   - **已确认**：Wan2.1-T2V-1.3B 的 `patch_size=[1,2,2]`（t=1, h=2, w=2），时间维 patch_size_t=1 对帧数无整除要求，**无需添加帧填充**。空间维 H、W 需被 2 整除，DOVE 训练分辨率 320×640 经 VAE 下采样后 80×160 满足。

2. **`initialize_pipeline` 与 `WanVideoPipeline` 构造**  
   - 使用 `WanVideoPipeline(device=..., torch_dtype=...)` 仅设基础参数，未传 `dit`、`vae` 等，当前通过后面 `pipe.dit = ...` 等赋值补全。  
   - WanVideoPipeline 的 `__init__` 未要求所有子模块必传，这种方式可行，但建议在文档中说明依赖 `from_pretrained` 的默认构造 + 后续赋值。

3. **`encode_text` 对 batch 的支持**  
   - 当前实现假定 `prompt` 为单个字符串。  
   - 在 `is_prompt_latent` 场景下，Dataset 对每个样本分别调用 `encode_text`，然后 collate 聚合，逻辑正确。  
   - 若未来支持 batch 调用，需扩展为 `Union[str, List[str]]` 并保持与 Dataset 行为一致。

4. **`validation_step` 中 `video` 的格式**  
   - `eval_data["video_tensor"]` 需满足 `[F, C, H, W]`，与 dove 一致。  
   - 先 `interpolate` 再 `frame_transform` 再 `permute` 到 `[1, C, F, H, W]` 的流程与 dove 一致，逻辑正确。

5. **`FlowMatchScheduler.add_noise` 前置条件**  
   - **已修复**：`add_noise` 依赖 `self.timesteps` 和 `self.sigmas`，必须在首次调用前执行 `scheduler.set_timesteps()`。  
   - 已在 `prepare_models` 中增加：`scheduler.set_timesteps(num_inference_steps=1000, denoising_strength=1.0)`，以支持任意 `noise_step` 的查找。  
   - `add_noise` 签名 `(original_samples, noise, timestep)` 与 DPM 一致，`torch.long` 的 timestep 会与 float 的 self.timesteps 兼容。

---

## 二、utils.py 修改对 CogVideoX 的影响

### 2.1 原 utils.py 设计（你提供的版本）

```python
SUPPORTED_MODELS: Dict[str, Dict[str, Trainer]] = {}

def register(model_name: str, training_type: Literal["lora", "sft"], trainer_cls: Trainer):
    if model_name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[model_name] = {}
    SUPPORTED_MODELS[model_name][training_type] = trainer_cls

def get_model_cls(model_type: str, training_type: Literal["lora", "sft"]) -> Trainer:
    return SUPPORTED_MODELS[model_type][training_type]
```

### 2.2 当前 utils.py 设计

```python
_REGISTRY = {}

def register(model_name: str, training_type: str, trainer_cls):
    key = f"{model_name}_{training_type}"
    _REGISTRY[key] = trainer_cls

def get_model_cls(model_name: str, training_type: str):
    key = f"{model_name}_{training_type}"
    return _REGISTRY[key]
```

### 2.3 结论：**不会影响 CogVideoX (dove) 的实现**

| 方面 | 说明 |
|------|------|
| **API 兼容性** | `register(model_name, training_type, trainer_cls)` 与 `get_model_cls(model_name, training_type)` 签名与语义均一致。 |
| **模型注册** | `dove/lora_one_s1_trainer.py` 中 `register("dove-s1", "lora", DOVES1Trainer)` 会写入 `_REGISTRY["dove-s1_lora"]`，与原 `SUPPORTED_MODELS["dove-s1"]["lora"]` 等价。 |
| **调用方式** | `train.py` 使用 `get_model_cls(args.model_name, args.training_type)`，两种实现都返回对应的 Trainer。 |
| **dove 与 dove-wan 共存** | 新设计用 `model_name_training_type` 作为 key，dove 与 dove-wan 的注册互不冲突。 |

原设计的 `show_supported_models()` 被移除，若需要调试可自行加回；对训练和推理流程无影响。

---

## 三、阶段一修改逻辑与步骤

### 3.1 修改原则

- 保持与 `model_fn_wan_video` 和 `WanVideoPipeline` 约定一致。  
- 不改动 dove 的 CogVideoX 实现，仅扩展 dove-wan。  
- 不改动 `utils.py` 的 `register`/`get_model_cls` 接口，保持对 CogVideoX 的兼容。

### 3.2 具体修改项

#### 修改 1：确认并处理 Wan latent 时间维度（如需）

**文件**: `finetune/models/dove-wan/lora_one_s1_trainer.py`

**目的**: 若 Wan DiT 对 `num_frames` 有整除要求，需做与 CogVideoX 类似的填充。

**操作**: 在 `compute_loss` 和 `validation_step` 中，检查 `model_fn_wan_video` 对 latent 形状的约束；若存在 `patch_size_t` 等要求，增加首帧复制逻辑（参考 dove 的 `patch_size_t` 处理）。

**代码示例**（仅在确认需要时添加）:

```python
# 在 compute_loss 中，编码后、送入 model_fn 前
patch_size_t = getattr(self.state.transformer_config, 'patch_size_t', None)
if patch_size_t is not None and lq_latent.shape[2] % patch_size_t != 0:
    ncopy = patch_size_t - (lq_latent.shape[2] % patch_size_t)
    first_frame = lq_latent[:, :, :1, :, :].repeat(1, 1, ncopy, 1, 1)
    lq_latent = torch.cat([first_frame, lq_latent], dim=2)
    hq_latent = torch.cat([first_frame[:,:,:ncopy], hq_latent], dim=2)  # 对应逻辑
```

#### 修改 2：`encode_text` 对 batch 的鲁棒性（可选）

**目的**: 避免未来在 batch prompt 场景出错。

**操作**: 若 `prompt` 为 `List[str]`，逐个或批量编码并 stack，保持输出形状 `[B, seq_len, hidden]`。当前 Dataset 仅传单条 prompt，可暂不实现。

#### 修改 3：`initialize_pipeline` 的明确性

**目的**: 明确 pipeline 的构造与填充方式。

**操作**: 在注释中说明依赖 `WanVideoPipeline` 的默认 `__init__`，再通过属性赋值注入 `dit/vae/tokenizer/text_encoder/scheduler`，与 DiffSynth 的 T2V 用法一致。

#### 修改 4：移除多余 import（如适用）

**目的**: 保持依赖干净。

**操作**: 若未使用 `FlowMatchScheduler`（仅在 pipeline 内部使用），可从 `lora_one_s1_trainer.py` 中删除对应 import。

---

## 四、新增 / 删除代码的目的与作用

| 类型 | 位置 | 内容 | 目的/作用 |
|------|------|------|-----------|
| **新增** | `load_components` | `sys.path.insert` 添加 DiffSynth-Studio | 保证能导入 `WanVideoPipeline`、`model_fn_wan_video` |
| **新增** | `load_components` | `ModelConfig` + `WanVideoPipeline.from_pretrained` | 按 T2V 最小配置加载 dit、vae、text_encoder |
| **新增** | `prepare_models` | 重写，不调用父类，构造 `transformer_config` | 规避 Wan VAE 不支持的接口，提供后续使用的 config |
| **新增** | `prepare_models` | `scheduler.set_timesteps(num_inference_steps=1000)` | FlowMatchScheduler 的 `add_noise` 依赖 timesteps/sigmas，必须在首次调用前初始化 |
| **新增** | `prepare_trainable_parameters` | 对 `nn.Module` 的 `requires_grad_` 检查 | 避免对 tokenizer 等非 Module 调用导致错误 |
| **新增** | `encode_video` | 转 list + `vae.encode(..., tiled=False)` | 匹配 Wan VAE 的 list 输入接口 |
| **新增** | `encode_text` | `(ids, mask)` + `seq_lens` 置零 padding | 匹配 Wan tokenizer 和 text encoder |
| **新增** | `compute_loss` | 使用 `model_fn_wan_video` 替代直接调用 transformer | 使用 Wan 的标准前向与 gradient checkpointing |
| **新增** | `validation_step` | 使用 `model_fn_wan_video` + `vae.decode` | 与 Wan 推理流程一致 |
| **删除** | `prepare_models` | 不再调用 `vae.enable_slicing/tiling` | Wan VAE 无此接口 |
| **删除** | `prepare_trainable_parameters` | 不再调用 `enable_gradient_checkpointing()` | 改为通过 model_fn 参数控制 |
| **删除** | `compute_loss` | 不再计算 RoPE、不再 permute latent | Wan DiT 内部处理 RoPE，且使用 [B,C,F,H,W] |

---

## 五、阶段一修改检查清单

- [x] **验证 `WanModel` 对 latent 帧数/尺寸的约束**：Wan2.1-T2V-1.3B `patch_size=[1,2,2]`，无需 patch_size_t 帧填充  
- [x] **修复 `FlowMatchScheduler.set_timesteps` 初始化**：在 `prepare_models` 中调用，确保 `add_noise` 可用（当 `noise_step != 0` 时必需）  
- [x] **确认 `add_noise` timestep dtype**：`torch.long` 与 FlowMatch 的 float timesteps 兼容  
- [ ] 在目标环境下跑通 Stage 1 训练若干 step（需用户执行）  
- [ ] 运行 validation_step，确认生成视频可正常解码与保存（需用户执行）  
- [ ] 确认 `dove-s1`（CogVideoX）训练与验证仍可正常执行（需用户执行）  

---

## 六、总结

- **dove-wan 的 `lora_one_s1_trainer.py` 整体设计合理**，与 WanVideoPipeline、`model_fn_wan_video` 的约定一致，能正确替换 CogVideoX backbone。  
- **当前 `utils.py` 修改不会影响 CogVideoX**，`register`/`get_model_cls` 的 API 与原有逻辑等价。  
- 建议优先完成 Wan 对 latent 形状的校验及 Stage 1 的端到端验证，其余修改按需进行。

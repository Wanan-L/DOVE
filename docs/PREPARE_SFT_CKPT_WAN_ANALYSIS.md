# prepare_sft_ckpt_wan.py 转换权重脚本验证分析

> 对照 CogVideoX 版 `prepare_sft_ckpt.py` 与 DiffSynth/Wan 结构，分析脚本逻辑正确性与潜在问题

---

## 一、脚本流程对比

### 1.1 CogVideoX (prepare_sft_ckpt) vs Wan (prepare_sft_ckpt_wan)

| 步骤 | CogVideoX | Wan | 差异说明 |
|------|-----------|-----|----------|
| **Step 1** | run_zero_to_fp32 | run_zero_to_fp32 | 相同：将 DeepSpeed checkpoint 转为 FP32 safetensors |
| **Step 2** | rename_weights | rename_weights_wan | Wan 版支持单文件与分片两种格式 |
| **Step 3** | prepare_ckpt_structure | prepare_wan_ckpt_structure | 目录结构不同（见下文） |
| **Step 4** | 清理 mid_output_dir | 同 | 相同 |

### 1.2 目录结构差异

| 项目 | CogVideoX | Wan |
|------|-----------|-----|
| **DiT 放置** | `ckpt/transformer/diffusion_pytorch_model*.safetensors` | `ckpt/diffusion_pytorch_model*.safetensors`（根目录） |
| **VAE / TextEncoder** | 在 weights_source 的 vae/、text_encoder/ 等子目录 | 理论上在同一根目录，但 DiffSynth 使用重定向路径 |
| **zero_to_fp32 输出** | model.safetensors（或 model-*.safetensors） | 同 |
| **重命名后** | diffusion_pytorch_model*.safetensors | 同 |

---

## 二、逻辑正确性分析

### 2.1 已正确实现的部分

1. **run_zero_to_fp32**
   - 调用方式与 CogVideoX 一致，使用 `--safe_serialization`
   - 依赖 `scripts/zero_to_fp32.py`，需在 `finetune/` 目录下运行

2. **rename_weights_wan**
   - 支持单文件：`model.safetensors` → `diffusion_pytorch_model.safetensors`
   - 支持分片：重命名 index 与分片文件，更新 weight_map
   - 比 CogVideoX 版更完备（CogVideoX 版假定总是分片）

3. **prepare_wan_ckpt_structure**
   - 先复制 weights_source 中除 DiT 外的所有文件
   - 再用训练得到的 DiT 权重替换 diffusion_pytorch_model*.safetensors
   - 与 Wan 的加载约定（根目录下 `diffusion_pytorch_model*.safetensors`）一致

### 2.2 潜在问题

#### 问题 1：weights_source 不完整（重要）

**现象**：默认 `weights_source = Wan-AI/Wan2.1-T2V-1.3B`，该目录下通常只有：

- `diffusion_pytorch_model.safetensors`（DiT，会被跳过）
- `google/umt5-xxl/`（tokenizer）

**缺失**：

- `models_t5_umt5-xxl-enc-bf16.safetensors`（TextEncoder）
- `Wan2.1_VAE.safetensors`（VAE）

这些文件实际位于 DiffSynth 的 redirect 路径，例如：

`DiffSynth-Studio/Wan-Series-Converted-Safetensors/`

**影响**：若只用默认 weights_source，生成的 ckpt 缺少 VAE 与 TextEncoder，后续加载会失败。

**建议**：

- 增加参数 `--weights_source_extra`，或
- 扩展逻辑：在复制基础文件时，从 redirect 路径补充 VAE 和 TextEncoder
- 或要求用户传入已包含 VAE、TextEncoder、tokenizer 的完整目录作为 weights_source

#### 问题 2：run_zero_to_fp32 的工作目录

**现象**：使用 `subprocess.run(["python3", "scripts/zero_to_fp32.py", ...])`，依赖当前工作目录。

**要求**：必须在 `finetune/` 下运行，才能找到 `scripts/zero_to_fp32.py`。

**建议**：在脚本开头根据 `__file__` 解析并切换到 `finetune/`，或对 `scripts/` 使用绝对路径。

#### 问题 3：dove-wan S2 未使用 model_path

**现象**：`lora_one_s2_trainer.py` 中 `load_components` 将 model_id 写死为 `"Wan-AI/Wan2.1-T2V-1.3B"`，未使用 `self.args.model_path`。

**影响**：即便 prepare_sft_ckpt_wan 生成了正确结构的 ckpt，Stage 2 仍会从预训练路径加载 DiT，不会用到转换后的 SFT 权重。这是 trainer 的问题，不是本脚本的问题。

**建议**：在 dove-wan S2 的 `load_components` 中，用 `model_path` 构建 DiT 的 `ModelConfig`（例如 `ModelConfig(path=model_path)` 或 `model_id=model_path`），以加载转换后的 ckpt。

---

## 三、与 CogVideoX 版的关键差异

| 项目 | CogVideoX | Wan |
|------|-----------|-----|
| **prepare_ckpt_structure** | 先 `copytree` 整个 weights_source，再清空 `transformer/` 并用训练权重填充 | 先复制 weights_source 中除 DiT 外的文件，再复制训练 DiT |
| **rename 单文件** | 无（假定分片） | 支持 |
| **weights_source 假设** | CogVideoX 目录结构完整 | Wan 使用 redirect，默认目录可能不完整 |

---

## 四、修改建议

### 4.1 高优先级

1. **保证 weights_source 完整**
   - 支持从 redirect 目录（如 `DiffSynth-Studio/Wan-Series-Converted-Safetensors`）复制 VAE、TextEncoder
   - 或增加 `--vae_path`、`--text_encoder_path` 等可选参数

2. **处理工作目录**
   - 在 `run_zero_to_fp32` 前显式切换到脚本所在目录的父目录（finetune/），或使用基于 `__file__` 的绝对路径调用 `zero_to_fp32.py`

### 4.2 中优先级

3. **单文件时的 rename 逻辑**
   - 当前单文件分支仅处理 `model.safetensors`；若 zero_to_fp32 输出其他命名，需要相应扩展

4. **补充错误处理**
   - 若 weights_source 中缺少必要文件，给出明确报错或提示

### 4.3 需与 trainer 配合的修改

5. **dove-wan S2 使用 model_path**
   - 在 `lora_one_s2_trainer.load_components` 中，对 DiT 使用 `model_path`，例如：

   ```python
   model_path = str(Path(self.args.model_path).resolve())
   model_configs = [
       ModelConfig(path=os.path.join(model_path, "diffusion_pytorch_model.safetensors"))
       if os.path.exists(os.path.join(model_path, "diffusion_pytorch_model.safetensors"))
       else ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
       # ... 其他 VAE、TextEncoder 配置
   ]
   ```

---

## 五、检查清单

- [x] run_zero_to_fp32 调用方式正确
- [x] rename_weights_wan 支持单文件与分片
- [x] prepare_wan_ckpt_structure 逻辑与 Wan 结构一致
- [x] **weights_source 完整性**：已添加 `--weights_source_extra` 及 DIFFSYNTH 环境变量自动推断
- [x] **工作目录**：已改为基于 `__file__` 的绝对路径调用，可从任意目录执行
- [x] **dove-wan S2**：已在 load_components 中使用 model_path 加载 DiT/VAE/TextEncoder/tokenizer
- [ ] 端到端验证：转换后的 ckpt 能被 Stage 2 正确加载（需用户执行）

---

## 六、总结与已实施修改

`prepare_sft_ckpt_wan.py` 的主体流程和目录设计合理。针对分析中的问题，已完成以下修改：

1. **prepare_sft_ckpt_wan.py**  
   - 工作目录：基于 `__file__` 使用绝对路径调用 `zero_to_fp32.py`，可在任意目录执行  
   - weights_source_extra：新增 `--weights_source_extra`，未指定时按 `DIFFSYNTH_MODEL_BASE_PATH` 自动推断，用于复制 VAE、TextEncoder  

2. **dove-wan lora_one_s2_trainer.py**  
   - 在 `load_components` 中优先从 `model_path` 加载 DiT、VAE、TextEncoder、tokenizer  
   - 若 `model_path` 下无对应文件，则回退到预训练 `Wan-AI/Wan2.1-T2V-1.3B`  

建议进行完整 Stage 1 → 转换 → Stage 2 的端到端验证。

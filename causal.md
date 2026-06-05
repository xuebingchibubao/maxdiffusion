# Causal-Forcing Framewise AR Inference Changes

本文档记录本次把 MaxDiffusion Wan2.1 T2V 推理路径改成 Causal-Forcing Stage 1 风格 framewise AR 推理的代码改动。

## 背景

原来的 `causal_forcing_framewise_ar.yml` 已经做了两件事：

1. 加载 Causal-Forcing Stage 1 权重：`/home/bennett121358/ar_diffusion.pt`
2. 在 Wan self-attention 中启用 `FramewiseCausalMask`

但旧推理路径仍然是 Wan 原始的 full-video parallel denoise：

```python
for step in range(num_inference_steps):
  pred = model(all_latent_frames, timestep=t)
  all_latent_frames = scheduler.step(pred, t, all_latent_frames)
```

Causal-Forcing Stage 1 AR 推理是按 latent frame/block 自回归生成：

```python
for frame_block in latent_frame_blocks:
  for t in scheduler.timesteps:
    pred = model(current_noisy_block, clean_history_cache, timestep=t)
    current_block = scheduler.step(pred, t, current_block)
  cache_clean_current_block(current_block, timestep=0)
```

## 本次实现

新增了 `framewise_ar_inference` 配置开关。开启后，Wan2.1 T2V 会走新的 framewise AR denoise loop：

```yaml
framewise_causal_attention: True
framewise_ar_inference: True
framewise_ar_num_frames_per_block: 1
framewise_ar_use_self_kv_cache: True
```

新的 loop 顺序是：

1. 保留初始完整 latent noise，形状为 `[B, C, F, H, W]`
2. 按 `framewise_ar_num_frames_per_block` 遍历 latent frame/block
3. 每个 block 单独重置 UniPC scheduler history
4. 对当前 block 跑完整 `num_inference_steps`
5. 只把当前 block 的 scheduler 结果写回 `generated_latents`
6. 已完成的历史 frame 在后续 block 中使用 timestep `0`，作为 clean context

当前实现新增了视频 self-attention KV cache。每个 frame/block denoise 完以后，会用 clean latent 和 timestep `0` 再跑一次 transformer，取出每一层 self-attention 的 K/V，并追加到历史 cache。后续 frame/block forward 只输入当前 latent frame/block，通过 `self_kv_cache` 读取过去 clean frames。

CFG 开启时，self KV cache 不再把 conditional/unconditional 拼成一个 `2 * batch` 的大 cache。当前实现把两路 CFG 分成两个半 batch forward：

```python
noise_cond = transformer(current_latents, prompt_cond_embeds, self_kv_cache=self_kv_cache_cond)
noise_uncond = transformer(current_latents, negative_prompt_embeds, self_kv_cache=self_kv_cache_uncond)
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

对应的 clean self KV cache 也分别维护：

```python
self_kv_cache_cond = append_self_kv_cache(self_kv_cache_cond, present_self_kv_cond)
self_kv_cache_uncond = append_self_kv_cache(self_kv_cache_uncond, present_self_kv_uncond)
```

这样可以避免在 HBM 上为单个 `2 * batch` cache buffer 做 5GB 级别的 concat 分配。

## 改动文件

### `src/maxdiffusion/pipelines/wan/wan_framewise_ar_utils.py`

新增 helper：

```python
def append_self_kv_cache(self_kv_cache, present_self_kv):
  if self_kv_cache is None:
    return present_self_kv
  return concatenate_along_sequence_axis(self_kv_cache, present_self_kv)
```

作用：

- 把当前 clean frame/block 产生的每层 self K/V 追加到历史 self K/V cache
- cache shape 约为 `[layers, batch, heads, cached_tokens, head_dim]`
- 追加轴是 sequence 轴

新增 helper：

```python
def make_framewise_ar_timestep(...):
  token_frame_ids = jnp.arange(seq_len) // tokens_per_frame
  current_frame_mask = (token_frame_ids >= frame_start) & (token_frame_ids < frame_start + num_frames)
  token_timestep = jnp.where(current_frame_mask, timestep, jnp.zeros((), dtype=jnp.asarray(timestep).dtype))
  return jnp.broadcast_to(token_timestep[None, :], (batch_size, seq_len))
```

作用：

- 构造 `[batch, seq_len]` 的 per-token timestep
- 当前正在 denoise 的 frame/block 使用当前 diffusion timestep
- 其他 frame 使用 `0`

新增 helper：

```python
def reset_scheduler_state_for_ar_block(scheduler_state):
  return scheduler_state.replace(
      model_outputs=jnp.zeros_like(scheduler_state.model_outputs),
      timestep_list=jnp.zeros_like(scheduler_state.timestep_list),
      lower_order_nums=0,
      last_sample=None,
      step_index=None,
      begin_index=None,
      this_order=0,
  )
```

作用：

- 每个 AR frame/block 都是独立的一条 denoise trajectory
- 因此进入新 frame/block 前必须清空 UniPC multistep history

新增 helper：

```python
def replace_latent_frame_block(latents, new_latents, frame_start, num_frames):
  if new_latents.shape[2] == num_frames:
    replacement = new_latents
  else:
    replacement = new_latents[:, :, frame_start : frame_start + num_frames]
  return latents.at[:, :, frame_start : frame_start + num_frames].set(
      replacement
  )
```

作用：

- 兼容两种输入：完整视频 latent tensor，或 KV cache 路径中的当前 frame/block tensor
- AR 推理只接受当前 frame/block 的更新
- 历史 clean frames 不再被后续 scheduler step 改写

### `src/maxdiffusion/pipelines/wan/wan_pipeline_2_1.py`

新增主函数：

```python
def run_framewise_ar_inference_2_1(...):
  self_kv_cache_cond = None
  self_kv_cache_uncond = None
  for frame_start in range(0, latent_num_frames, num_frames_per_block):
    current_latents = generated_latents[:, :, frame_start : frame_start + 1]

    for step in range(num_inference_steps):
      noise_cond = transformer(current_latents, prompt_cond_embeds, self_kv_cache=self_kv_cache_cond, timestep=t)
      noise_uncond = transformer(current_latents, negative_prompt_embeds, self_kv_cache=self_kv_cache_uncond, timestep=t)
      noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
      current_latents, frame_scheduler_state = scheduler.step(...)

    present_self_kv_cond = transformer(..., return_self_kv=True)
    present_self_kv_uncond = transformer(..., return_self_kv=True)
    self_kv_cache_cond = append_self_kv_cache(self_kv_cache_cond, present_self_kv_cond)
    self_kv_cache_uncond = append_self_kv_cache(self_kv_cache_uncond, present_self_kv_uncond)
    generated_latents = replace_latent_frame_block(...)

  return generated_latents
```

作用：

- 实现 Causal-Forcing 风格的外层 AR frame loop
- 保留已有 CFG 逻辑
- 复用已有 `transformer_forward_pass_full_cfg`
- 复用已有 UniPC scheduler
- 要求 `framewise_causal_attention=True`
- 不兼容 `use_cfg_cache=True` 和 `use_magcache=True`
- `framewise_ar_use_self_kv_cache=True` 时要求 `framewise_ar_num_frames_per_block=1`

在 `run_inference_2_1` 中新增分支：

```python
if getattr(config, "framewise_ar_inference", False):
  return run_framewise_ar_inference_2_1(...)
```

未开启 `framewise_ar_inference` 时，原 Wan2.1 并行 denoise 路径不变。

### `src/maxdiffusion/configs/causal_forcing_framewise_ar.yml`

新增：

```yaml
framewise_ar_inference: True
framewise_ar_num_frames_per_block: 1
framewise_ar_use_self_kv_cache: True
```

同时保留：

```yaml
framewise_causal_attention: True
```

并对齐 Causal-Forcing framewise AR 配置：

```yaml
flow_shift: 5.0
num_inference_steps: 50
guidance_scale: 3.0
```

### `src/maxdiffusion/tests/wan/wan_framewise_ar_inference_test.py`

新增单测覆盖：

1. 当前 frame 的 tokens 使用当前 diffusion timestep
2. 多 frame block 的 timestep map 正确
3. scheduler 输出只写回当前 frame/block
4. self KV cache 沿 sequence 轴追加

### `src/maxdiffusion/models/attention_flax.py`

Tokamax/Splash block-size 选择新增了 128 lane 对齐和 cache 场景的保守分块：

```python
if attention_kernel in ["tokamax_flash", "tokamax_ring"] and key_seq_len != query_seq_len:
  block_size_q = round_up_to_multiple(configured_block_q, 128)
  block_kv_size = min(round_up_to_multiple(configured_block_kv, 128), padded_key_seq_len)
  block_kv_compute_size = aligned_divisor_at_most(block_kv_size, configured_block_kv_compute)
```

作用：

- 解决 `bkv_compute=3120 must be a multiple of 128`
- 解决 KV cache 增长后 `block_kv=11008` 导致的 Tokamax compile-time vmem OOM
- 对当前配置保持 `block_kv=512`、`block_kv_compute=512`，让长 KV cache 继续按 512 分块计算

`FlaxWanAttention.__call__` 新增：

```python
cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None
return_kv: bool = False
```

self-attention 中：

```python
present_kv = (key_proj, value_proj)
if cached_kv is not None and "self" in cached_kv:
  cached_key, cached_value = cached_kv["self"]
  key_proj = jnp.concatenate([cached_key, key_proj], axis=2)
  value_proj = jnp.concatenate([cached_value, value_proj], axis=2)
```

作用：

- `present_kv` 保存当前 frame/block 的 RoPE 后 self K/V
- 后续 frame/block 把历史 K/V 拼到当前 K/V 前面
- self KV cache 和原有 cross-attention text KV cache 使用不同参数传递，避免混淆

### `src/maxdiffusion/models/wan/transformers/transformer_wan.py`

`WanTransformerBlock.__call__` 新增：

```python
self_kv_cache=None
return_self_kv=False
```

`WanModel.__call__` 新增：

```python
self_kv_cache=None
return_self_kv=False
```

作用：

- 每层 block 从 `self_kv_cache` 读取对应层的历史 self K/V
- 当 `return_self_kv=True` 时，把当前 frame/block 的每层 self K/V stack 成 cache tree 返回
- 原有 `kv_cache` 仍只用于 cross-attention text/image K/V

## 当前限制

这次实现的是 Causal-Forcing-style AR 推理语义，并加入了视频 self-attention KV cache，但还不是官方最高效版本。

差别：

- 官方 PyTorch 实现使用固定容量/append 式视频 self-attention KV cache
- 当前 MaxDiffusion 实现使用增长式 JAX pytree cache，cache sequence length 会随 frame 增长
- 因此可能按不同 cache length 触发多次 XLA 编译
- 但后续 denoise forward 已经只输入当前 latent frame，不再重算历史 latent frame 的 QKV

以 81 帧视频为例：

- VAE latent frames: 21
- 默认 denoise steps: 50
- denoise forward 约 1050 次，但每次只处理当前 latent frame 的 query
- 额外 cache update forward 约 21 次，用于把 clean frame 写入 self KV cache

后续如果要进一步接近官方速度，需要把增长式 cache 改成固定 shape cache，避免每个 cache length 产生新的编译形状。

## 验证

已新增 focused unit test：

```bash
python -m pytest src/maxdiffusion/tests/wan/wan_framewise_ar_inference_test.py
```

建议同时跑已有 mask test：

```bash
python -m pytest src/maxdiffusion/tests/wan/wan_framewise_mask_test.py
```

本次还验证了 attention block-size 选择：

```bash
python -m pytest src/maxdiffusion/tests/attention_test.py -k "select_flash_block_sizes or tokamax_cached_self_attention"
```

当前实测：

- `wan_framewise_ar_inference_test.py` + `wan_framewise_mask_test.py`: 7 passed
- attention block-size focused tests: 5 passed

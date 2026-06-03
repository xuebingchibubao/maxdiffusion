# Framewise Causal Video Generation 实现过程

本文档记录本次 toy task 的主要代码改动：

- 在 MaxDiffusion 中跑通 Causal Forcing Stage1 framewise AR diffusion。
- 将 Wan 视频 self-attention 从双向注意力改成 framewise causal attention。
- 使用 MaxDiffusion 现有的 Splash/Tokamax Splash Attention 路径承载 framewise mask。
- 加载本地 Causal Forcing checkpoint：`/home/bennett121358/ar_diffusion.pt`。

## 运行入口

新增的运行配置文件是：

```bash
src/maxdiffusion/configs/causal_forcing_framewise_ar.yml
```

示例运行命令：

```bash
python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/causal_forcing_framewise_ar.yml \
  prompt="A small toy robot walking across a desk" \
  num_inference_steps=4
```

这个配置使用 Wan2.1 1.3B Diffusers 的 VAE、text encoder 和 scheduler，同时把 transformer 权重替换成本地 Causal Forcing 权重：

```yaml
wan_transformer_pretrained_model_name_or_path: '/home/bennett121358/ar_diffusion.pt'
attention: 'tokamax_flash'
framewise_causal_attention: True
```

## 从双向注意力改成 Framewise Causal

MaxDiffusion 原本的 Wan self-attention 是双向的。也就是说，所有 latent video token 之间可以互相 attend：

```text
任意 video token -> 任意 video token
```

Causal Forcing 的 framewise AR diffusion 需要的是按帧自回归的注意力。Wan patch embedding 后，video token 是按 frame-major 顺序排列的：

```text
frame 0 的所有 token, frame 1 的所有 token, frame 2 的所有 token, ...
```

本次实现的 causal 规则是“按帧 causal”，不是普通 token-level causal：

```text
query frame t 可以 attend 到 <= t 的所有视频帧
query frame t 不能 attend 到 > t 的未来视频帧
同一帧内部的所有空间 token 可以互相 attend
```

这一点很重要。普通 causal attention 通常是严格的下三角 token mask，它会屏蔽同一帧里排在当前 token 后面的空间 token；但 framewise causal 不应该这么做。同一帧内部应该是全可见的，只有未来帧需要被屏蔽。

## Framewise Mask 实现

新增 mask 类的位置：

```text
src/maxdiffusion/kernels/splash_attention/splash_attention_mask.py
```

新增类：

```python
FramewiseCausalMask
```

核心逻辑：

```python
q_frame = q_ids // tokens_per_frame
kv_frame = kv_ids // tokens_per_frame
return q_frame >= kv_frame
```

其中 `tokens_per_frame` 表示 patch 后每一帧包含多少 latent token。

在 WanModel 里，这个值通过 patch 后的空间尺寸得到：

```python
tokens_per_frame = post_patch_height * post_patch_width
```

对应代码位置：

```text
src/maxdiffusion/models/wan/transformers/transformer_wan.py
```

当配置里启用：

```yaml
framewise_causal_attention: True
```

WanModel 会构造：

```python
self_attention_mask = FramewiseCausalMask(
    shape=(hidden_states.shape[1], hidden_states.shape[1]),
    tokens_per_frame=tokens_per_frame,
)
```

然后把这个 mask 传给每一层 Wan transformer block 的 self-attention。

## 接入 Splash Attention

MaxDiffusion 当前已有两类相关 attention kernel：

- JAX/Pallas 提供的 Splash Attention：
  `jax.experimental.pallas.ops.tpu.splash_attention`
- MaxDiffusion 本地的 Tokamax Splash Attention：
  `src/maxdiffusion/kernels/splash_attention`

本次配置使用的是：

```yaml
attention: 'tokamax_flash'
```

主要改动位置：

```text
src/maxdiffusion/models/attention_flax.py
```

这里新增了一层 mask 类型区分：

```text
batch padding mask:
  shape 类似 [batch, kv_len]
  用于 padding token 的 segment ids

semantic QxKV mask:
  shape 为 [q_len, kv_len]
  用于真正的 attention 可见性控制
```

Framewise causal mask 属于第二类，也就是 semantic QxKV mask。

原来的 cross-attention padding mask 行为仍然保留；新增逻辑只是在 self-attention 开启 framewise causal 时，把二维 mask 作为 Splash mask 传入 kernel。

Tokamax Splash 内部会对 sequence length 做 padding，所以 mask 的 shape 必须和 padding 后的 Q/KV 长度一致。因此对于 `FramewiseCausalMask`，代码会在进入 Tokamax kernel 前按 padding 后 shape 重建一次：

```python
FramewiseCausalMask(
    shape=(padded_q_len, padded_kv_len),
    tokens_per_frame=attention_mask.tokens_per_frame,
)
```

这样可以避免 mask shape 和 kernel 输入 shape 不一致。

对于非 Tokamax 或 dot-product fallback，代码会把 semantic mask materialize 成 dense bool mask，并在 softmax 前把不可见位置设成很小的 mask value。

## Mask 在 Wan 中的传递路径

原始 Wan self-attention 调用大致是：

```python
self.attn1(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_hidden_states,
    rotary_emb=rotary_emb,
)
```

现在新增传入：

```python
self_attention_mask=self_attention_mask
```

`FlaxWanAttention` 中也记录了这个 attention 层是否是 self-attention：

```python
self.is_self_attention = is_self_attention
```

这么做是因为 Wan self-attention 的调用里显式传了：

```python
encoder_hidden_states=norm_hidden_states
```

如果仅靠 `encoder_hidden_states is None` 判断 self/cross attention，会把 self-attention 误判成 cross-attention。现在使用初始化时的 `is_self_attention` 标记，确保 framewise mask 只作用在 self-attention 上。

## 加载 Causal Forcing 模型参数

Causal Forcing 权重是本地 PyTorch `.pt` 文件：

```text
/home/bennett121358/ar_diffusion.pt
```

MaxDiffusion 原来的 Wan loader 主要支持：

- Diffusers safetensors 目录
- 已知的 CausVid/FusionX 模型名

因此本次新增了本地 `.pt` checkpoint 的加载逻辑。

代码位置：

```text
src/maxdiffusion/models/wan/wan_utils.py
```

新增函数：

```python
load_causal_forcing_transformer(...)
```

它会执行：

```python
torch.load(checkpoint_path, map_location="cpu")
```

然后从 checkpoint 中选择 state dict。支持的 key 包括：

```text
generator
generator_ema
state_dict
model
```

拿到 PyTorch state dict 后，复用 MaxDiffusion 已有的 Wan key 转换路径：

```python
rename_key(...)
rename_for_custom_trasformer(...)
get_key_and_value(...)
```

其中有一个关键命名差异需要特殊处理。

Causal Forcing 的 block modulation 经过旧的/custom key mapping 后会变成：

```text
blocks.*.scale_shift_table
```

但当前 MaxDiffusion Wan block 里的真实参数名是：

```text
blocks.*.adaln_scale_shift_table
```

所以 loader 中增加了重写：

```python
.scale_shift_table -> .adaln_scale_shift_table
```

否则会出现类似：

```text
KeyError: ('blocks', 'scale_shift_table')
```

## 允许本地 .pt 路径

MaxDiffusion 的 `pyconfig` 原来会拒绝未知的 Wan transformer path。

为了让配置可以使用：

```yaml
wan_transformer_pretrained_model_name_or_path: '/home/bennett121358/ar_diffusion.pt'
```

在这里增加了本地 `.pt` 白名单逻辑：

```text
src/maxdiffusion/pyconfig.py
```

当路径满足：

```python
path.endswith(".pt") and os.path.isfile(path)
```

就允许作为 Wan2.1 transformer 权重，并默认启用：

```python
framewise_causal_attention = True
```

## 配置文件改动

新增配置文件：

```text
src/maxdiffusion/configs/causal_forcing_framewise_ar.yml
```

它基于：

```text
src/maxdiffusion/configs/base_wan_1_3b.yml
```

主要修改项：

```yaml
run_name: 'causal-forcing-framewise-ar'
pretrained_model_name_or_path: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
wan_transformer_pretrained_model_name_or_path: '/home/bennett121358/ar_diffusion.pt'
attention: 'tokamax_flash'
framewise_causal_attention: True
guidance_scale: 3.0
output_dir: 'causal-forcing-framewise-ar-output'
```

同时补齐 `generate_wan.py` 会读取的 Wan2.1 cache 字段：

```yaml
use_cfg_cache: False
use_kv_cache: False
use_magcache: False
magcache_thresh: 0.12
magcache_K: 2
retention_ratio: 0.2
```

否则会出现：

```text
AttributeError: Requested key use_magcache, not in config
```

## 测试与验证

新增 focused test：

```text
src/maxdiffusion/tests/wan/wan_framewise_mask_test.py
```

测试内容：

- frame 0 只能 attend frame 0
- frame 1 可以 attend frame 0 和 frame 1
- frame 2 可以 attend frame 0、frame 1 和 frame 2
- 同一帧内部 token 互相可见

运行命令：

```bash
PYTHONPATH=src /home/bennett121358/maxdiffusion_venv/bin/python -m pytest \
  src/maxdiffusion/tests/wan/wan_framewise_mask_test.py -q
```

结果：

```text
2 passed
```

还做过：

```bash
/home/bennett121358/maxdiffusion_venv/bin/python -m py_compile ...
git diff --check
```

## 当前 TPU 运行进展

在 TPU VM 上运行时，已经到达：

```text
load_time: ...
compile_time: ...
```

这说明以下步骤已经通过：

- 当前仓库代码被正确导入
- Causal Forcing 配置被接受
- Wan VAE、text encoder、scheduler 能加载
- `/home/bennett121358/ar_diffusion.pt` transformer 权重能加载
- TPU compile 能启动并完成

后续会进入真正推理、VAE decode 和视频导出阶段。

## Bring-up 过程中遇到的问题

### 1. 没有使用本地源码

一开始 Python 导入的是虚拟环境里安装过的旧版 MaxDiffusion：

```text
site-packages/maxdiffusion/pyconfig.py
```

而不是当前仓库的：

```text
src/maxdiffusion/pyconfig.py
```

解决方式：

```bash
pip install -e .
```

或者临时使用：

```bash
PYTHONPATH=src
```

### 2. 缺少 numpy import

新增的 `attention_flax.py` 逻辑用了 `np.ndarray`，但文件原来没有：

```python
import numpy as np
```

已补上。

### 3. block modulation key 映射错误

报错：

```text
KeyError: ('blocks', 'scale_shift_table')
```

原因是当前 MaxDiffusion 使用：

```text
adaln_scale_shift_table
```

已在 Causal Forcing `.pt` loader 中修复。

### 4. 配置缺少 use_magcache

报错：

```text
AttributeError: Requested key use_magcache, not in config
```

原因是 `generate_wan.py` 会读取该字段，但 `base_wan_1_3b.yml` 没有。

已在 `causal_forcing_framewise_ar.yml` 中补齐相关 cache 字段。

### 5. CUDA cuInit 日志

日志里可能出现：

```text
failed call to cuInit
```

当前任务运行在 TPU 上，JAX/TF 可能仍会探测 CUDA。这个日志不是本任务失败原因。

### 6. transformer_engine 缺失日志

可能看到：

```text
ModuleNotFoundError: No module named 'transformer_engine'
```

这是 optional Transformer Engine context manager 相关日志。实际 bring-up 中导致退出的主错误不是它，而是上面几个配置或权重映射问题。

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp


def make_framewise_ar_timestep(
    timestep: jax.Array,
    batch_size: int,
    seq_len: int,
    tokens_per_frame: int,
    frame_start: int,
    num_frames: int,
) -> jax.Array:
  """Build a [B, seq_len] timestep map for framewise AR denoising."""
  token_frame_ids = jnp.arange(seq_len) // tokens_per_frame
  current_frame_mask = (token_frame_ids >= frame_start) & (token_frame_ids < frame_start + num_frames)
  token_timestep = jnp.where(current_frame_mask, timestep, jnp.zeros((), dtype=jnp.asarray(timestep).dtype))
  return jnp.broadcast_to(token_timestep[None, :], (batch_size, seq_len))


def reset_scheduler_state_for_ar_block(scheduler_state):
  """Reset UniPC history before denoising a new autoregressive frame block."""
  return scheduler_state.replace(
      model_outputs=jnp.zeros_like(scheduler_state.model_outputs),
      timestep_list=jnp.zeros_like(scheduler_state.timestep_list),
      lower_order_nums=0,
      last_sample=None,
      step_index=None,
      begin_index=None,
      this_order=0,
  )


def replace_latent_frame_block(latents: jax.Array, new_latents: jax.Array, frame_start: int, num_frames: int):
  if new_latents.shape[2] == num_frames:
    replacement = new_latents
  else:
    replacement = new_latents[:, :, frame_start : frame_start + num_frames]
  return latents.at[:, :, frame_start : frame_start + num_frames].set(replacement)


def append_self_kv_cache(self_kv_cache, present_self_kv):
  if self_kv_cache is None:
    return present_self_kv

  appended = {}
  for key in present_self_kv:
    cached_key, cached_value = self_kv_cache[key]
    present_key, present_value = present_self_kv[key]
    appended[key] = (
        jnp.concatenate([cached_key, present_key], axis=3),
        jnp.concatenate([cached_value, present_value], axis=3),
    )
  return appended

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

import unittest

import jax

jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import numpy as np

from maxdiffusion.pipelines.wan.wan_framewise_ar_utils import (
    append_self_kv_cache,
    make_framewise_ar_timestep,
    replace_latent_frame_block,
)


class WanFramewiseArInferenceTest(unittest.TestCase):

  def test_make_framewise_ar_timestep_marks_only_current_frame(self):
    timestep = make_framewise_ar_timestep(
        timestep=jnp.array(7, dtype=jnp.int32),
        batch_size=2,
        seq_len=20,
        tokens_per_frame=4,
        frame_start=2,
        num_frames=1,
    )

    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    np.testing.assert_array_equal(np.asarray(timestep), expected)

  def test_make_framewise_ar_timestep_supports_frame_blocks(self):
    timestep = make_framewise_ar_timestep(
        timestep=jnp.array(11, dtype=jnp.int32),
        batch_size=1,
        seq_len=20,
        tokens_per_frame=4,
        frame_start=1,
        num_frames=2,
    )

    expected = np.array([[0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(np.asarray(timestep), expected)

  def test_replace_latent_frame_block_updates_only_current_block(self):
    latents = jnp.zeros((1, 1, 4, 1, 1), dtype=jnp.float32)
    new_latents = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 4, 1, 1)

    updated = replace_latent_frame_block(latents, new_latents, frame_start=1, num_frames=2)

    expected = np.array([0, 1, 2, 0], dtype=np.float32).reshape(1, 1, 4, 1, 1)
    np.testing.assert_array_equal(np.asarray(updated), expected)

  def test_replace_latent_frame_block_accepts_current_block_tensor(self):
    latents = jnp.zeros((1, 1, 4, 1, 1), dtype=jnp.float32)
    new_latents = jnp.full((1, 1, 1, 1, 1), 5.0, dtype=jnp.float32)

    updated = replace_latent_frame_block(latents, new_latents, frame_start=2, num_frames=1)

    expected = np.array([0, 0, 5, 0], dtype=np.float32).reshape(1, 1, 4, 1, 1)
    np.testing.assert_array_equal(np.asarray(updated), expected)

  def test_append_self_kv_cache_appends_sequence_axis(self):
    cache = {
        "self": (
            jnp.zeros((2, 1, 3, 4, 5), dtype=jnp.float32),
            jnp.ones((2, 1, 3, 4, 5), dtype=jnp.float32),
        )
    }
    present = {
        "self": (
            jnp.full((2, 1, 3, 2, 5), 2.0, dtype=jnp.float32),
            jnp.full((2, 1, 3, 2, 5), 3.0, dtype=jnp.float32),
        )
    }

    updated = append_self_kv_cache(cache, present)

    self.assertEqual(updated["self"][0].shape, (2, 1, 3, 6, 5))
    self.assertEqual(updated["self"][1].shape, (2, 1, 3, 6, 5))
    np.testing.assert_array_equal(np.asarray(updated["self"][0][:, :, :, :4]), np.zeros((2, 1, 3, 4, 5)))
    np.testing.assert_array_equal(np.asarray(updated["self"][0][:, :, :, 4:]), np.full((2, 1, 3, 2, 5), 2.0))


if __name__ == "__main__":
  unittest.main()

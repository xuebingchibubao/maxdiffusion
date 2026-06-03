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

import numpy as np

from maxdiffusion.kernels.splash_attention import splash_attention_mask as mask_lib


class WanFramewiseMaskTest(unittest.TestCase):

  def test_framewise_causal_mask_allows_current_frame_and_history(self):
    mask = mask_lib.FramewiseCausalMask(shape=(6, 6), tokens_per_frame=2)

    expected = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(mask[:, :], expected)

  def test_framewise_causal_mask_supports_slices(self):
    mask = mask_lib.FramewiseCausalMask(shape=(8, 8), tokens_per_frame=2)

    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(mask[2:6, 0:4], expected)


if __name__ == "__main__":
  unittest.main()

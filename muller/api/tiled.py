# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/api/tiled.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import Optional, Tuple, Union

import numpy as np

from muller.core.partial_sample import PartialSample


def tiled(
    sample_shape: Tuple[int, ...],
    tile_shape: Optional[Tuple[int, ...]] = None,
    dtype: Union[str, np.dtype] = np.dtype("uint8"),
):
    """Allocates an empty sample of shape ``sample_shape``, broken into tiles of shape ``tile_shape``
        (except for edge tiles).

    Example:

        >>> with ds:
        ...    ds.create_tensor("image", htype="image", sample_compression="png")
        ...    ds.image.append(muller.tiled(sample_shape=(1003, 1103, 3), tile_shape=(10, 10, 3)))
        ...    ds.image[0][-217:, :212, 1:] = np.random.randint(0, 256, (217, 212, 2), dtype=np.uint8)

    Args:
        sample_shape (Tuple[int, ...]): Full shape of the sample.
        tile_shape (Optional, Tuple[int, ...]): The sample will be will stored as tiles where each tile will have this
            shape (except edge tiles). If not specified, it will be computed such that each tile is close to half of the
            tensor's `max_chunk_size` (after compression).
        dtype (Union[str, np.dtype]): Dtype for the sample array. Default uint8.

    Returns:
        PartialSample: A PartialSample instance which can be appended to a Tensor.
    """
    return PartialSample(sample_shape=sample_shape, tile_shape=tile_shape, dtype=dtype)

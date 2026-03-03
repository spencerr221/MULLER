# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin


def size_approx(dataset):
    """Estimates the size in bytes of the dataset.

    Args:
        dataset: The dataset to estimate size for.

    Returns:
        int: Estimated size in bytes.
    """
    tensors = dataset.version_state["full_tensors"].values()
    chunk_engines = [tensor.chunk_engine for tensor in tensors]
    size = sum(c.num_chunks * c.min_chunk_size for c in chunk_engines)
    return size

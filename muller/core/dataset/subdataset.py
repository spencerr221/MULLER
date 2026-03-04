# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Subdataset loading utilities for Dataset."""

from typing import Optional

import muller.core.dataset
from muller.constants import DEFAULT_LOCAL_CACHE_SIZE, DEFAULT_MEMORY_CACHE_SIZE, MB
from muller.core.storage.cache_chain import generate_chain


def sub_ds(
        dataset,
        path,
        empty=False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        read_only=None,
        verbose=True,
):
    """Loads a nested dataset. Internal."""
    sub_storage = dataset.base_storage.subdir(path, read_only=read_only)

    if empty:
        sub_storage.clear()

    path = sub_storage
    cls = muller.core.dataset.Dataset

    ret = cls(
        generate_chain(
            sub_storage,
            memory_cache_size * MB,
            local_cache_size * MB,
        ),
        path=path,
        read_only=read_only,
        verbose=verbose,
    )
    ret.parent_dataset = dataset
    return ret

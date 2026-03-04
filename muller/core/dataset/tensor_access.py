# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Tensor access and filtering utilities for Dataset."""

from typing import Dict, List

from muller.constants import VDS_INDEX
from muller.core.tensor import Tensor
from muller.util.exceptions import InvalidKeyTypeError


def get_tensors(dataset, include_hidden: bool = True, include_disabled=True) -> Dict[str, Tensor]:
    """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
    version_state = dataset.version_state
    index = dataset.index
    all_tensors = all_tensors_filtered(dataset, include_hidden, include_disabled)
    return {
        t: version_state["full_tensors"][
            version_state["tensor_names"][t]
        ][index]
        for t in all_tensors
    }


def all_tensors_filtered(dataset, include_hidden: bool = True, include_disabled=True) -> List[str]:
    """Names of all tensors belonging to this group, including those within sub groups"""
    hidden_tensors = dataset.meta.hidden_tensors
    tensor_names = dataset.version_state["tensor_names"]
    enabled_tensors = dataset.enabled_tensors
    final_results = []
    for t in tensor_names:
        if include_hidden or tensor_names[t] not in hidden_tensors:
            if include_disabled or enabled_tensors is None or t in enabled_tensors:
                final_results.append(t)
    return final_results


def get_sample_indices(dataset, maxlen: int):
    """Get sample indices"""
    vds_index = get_tensors(dataset, include_hidden=True).get(VDS_INDEX)
    if vds_index:
        return vds_index.numpy().reshape(-1).tolist()
    return dataset.index.values[0].indices(maxlen)


def resolve_tensor_list(dataset, keys: List[str]) -> List[str]:
    """Resolve the tensor list."""
    ret = []
    for k in keys:
        fullpath = k
        if (
                dataset.version_state["tensor_names"].get(fullpath)
                in dataset.version_state["full_tensors"]
        ):
            ret.append(k)
        else:
            enabled_tensors = dataset.enabled_tensors
            if fullpath[-1] != "/":
                fullpath = fullpath + "/"
            hidden = dataset.meta.hidden_tensors
            for temp_tensor in dataset.version_state["tensor_names"]:
                temp_tensor_valid = temp_tensor.startswith(fullpath) and temp_tensor not in hidden
                if temp_tensor_valid and (enabled_tensors is None or temp_tensor in enabled_tensors):
                    ret.append(temp_tensor)
    return ret

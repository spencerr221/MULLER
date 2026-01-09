# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Union, Any

from muller.core.storage.muller_memory_object import MULLERMemoryObject


def _get_nbytes(obj: Union[bytes, memoryview, MULLERMemoryObject]):
    if isinstance(obj, MULLERMemoryObject):
        return obj.nbytes
    return len(obj)


def identity(x: Any):
    """Identity function."""
    return x

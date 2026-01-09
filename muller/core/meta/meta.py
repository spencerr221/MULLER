# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any, Dict
from muller.core.storage.muller_memory_object import MULLERMemoryObject
import muller


class Meta(MULLERMemoryObject):
    """Contains **required** key/values that datasets/tensors use to function.
    See the ``Info`` class for optional key/values for datasets/tensors.
    """

    def __init__(self):
        super().__init__()

        self.version = muller.__version__

    def __getstate__(self) -> Dict[str, Any]:
        return {"version": self.version}

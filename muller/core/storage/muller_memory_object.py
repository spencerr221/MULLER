# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
import json
from typing import Any, Dict
from muller.util.json import HubJsonDecoder, HubJsonEncoder


class MULLERMemoryObject(ABC):
    def __init__(self):
        self.is_dirty = True

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    @property
    @abstractmethod
    def nbytes(self):
        """Returns the number of bytes in the object."""
    @classmethod
    def frombuffer(cls, buffer: bytes, **kwargs):
        """Function to read from buffer."""
        instance = cls()
        if len(buffer) > 0:
            instance.__setstate__(json.loads(buffer, cls=HubJsonDecoder))
            instance.is_dirty = False
            return instance
        raise BufferError(f"Unable to instantiate the object as the buffer was empty.{buffer},{type(buffer)}")

    def tobytes(self) -> bytes:
        """Function to convert to bytes."""
        d = {str(k): v for k, v in self.__getstate__().items()}
        return bytes(json.dumps(d, sort_keys=True, indent=4, cls=HubJsonEncoder), "utf-8")

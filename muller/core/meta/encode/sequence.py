# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from muller.core.meta.encode.byte_positions import BytePositionsEncoder
from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.serialize import (
    serialize_sequence_or_creds_encoder,
    deserialize_sequence_or_creds_encoder,
)


class SequenceEncoder(BytePositionsEncoder, MULLERMemoryObject):
    @classmethod
    def frombuffer(cls, buffer: bytes, **kwargs):
        instance = cls()
        if not buffer:
            return instance

        version, ids = deserialize_sequence_or_creds_encoder(buffer, "seq")
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        instance.is_dirty = False
        return instance

    def tobytes(self) -> memoryview:
        return memoryview(
            serialize_sequence_or_creds_encoder(self.version, self._encoded)
        )

    def pop(self, index):
        self.is_dirty = True
        super().pop(index)

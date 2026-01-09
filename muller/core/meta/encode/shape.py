# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Tuple

import numpy as np

from muller.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN


class ShapeEncoder(Encoder):

    @property
    def dimensionality(self) -> int:
        """Function to get the dimensionality."""
        return len(self[0])

    def _combine_condition(
        self, item: Tuple[int], compare_row_index: int = -1
    ) -> bool:
        last_shape = self._derive_value(self._encoded[compare_row_index])

        return item == last_shape

    def _derive_value(self, row: np.ndarray, *_) -> Tuple:  # type: ignore
        return tuple(row[:LAST_SEEN_INDEX_COLUMN])

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/shape_interval.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.util.exceptions import InvalidShapeIntervalError
from typing import Optional, Sequence, Tuple


def _contains_negatives(shape: Sequence[int]):
    return any(x and x < 0 for x in shape)


class ShapeInterval:
    def __init__(self, lower: Sequence[int], upper: Optional[Sequence[int]] = None):
        if upper is None:
            upper = lower

        if len(lower) != len(upper):
            raise InvalidShapeIntervalError("Lengths must match.", lower, upper)

        if _contains_negatives(lower):
            raise InvalidShapeIntervalError("Lower cannot contain negative components.", lower=lower)

        if _contains_negatives(upper):
            raise InvalidShapeIntervalError("upper cannot contain negative components.", upper=upper)

        if not all(l is None or u is None or l <= u for l, u in zip(lower, upper)):
            raise InvalidShapeIntervalError("lower[i] must always be <= upper[i].", lower=lower, upper=upper)

        self._lower = tuple(lower)
        self._upper = tuple(upper)

    def astuple(self) -> Tuple[Optional[int], ...]:
        # TODO: named tuple? NHWC shape would be (10, 224, 224, 3) could be (N=10, H=224, W=224, C=3).

        shape = []
        for low, up in zip(self.lower, self.upper):
            shape.append(None if low != up else low)  # type: ignore
        return tuple(shape)

    @property
    def is_dynamic(self) -> bool:
        return self.lower != self.upper

    @property
    def lower(self) -> Tuple[int, ...]:
        return self._lower

    @property
    def upper(self) -> Tuple[int, ...]:
        return self._upper

    def __str__(self):
        intervals = []

        for l, u in zip(self.lower, self.upper):
            if l == u:
                intervals.append(str(l))
            else:
                intervals.append(f"{l}:{u}")

        if len(intervals) == 1:
            return f"({intervals[0]},)"
        return f"({', '.join(intervals)})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, ShapeInterval):
            return False

        return self.lower == other.lower and self.upper == other.upper

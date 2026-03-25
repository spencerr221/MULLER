# SPDX-License-Identifier: MPL-2.0
#
# Copyright (c) 2026 Xueling Lin

"""HubJsonEncoder must convert numpy scalars; returning them unchanged breaks json.dumps
with ValueError: Circular reference detected."""

import json

import numpy as np

from muller.util.json import HubJsonEncoder


def test_numpy_scalars_encode_without_circular_reference():
    payload = {
        "nan_count": np.int64(3),
        "nan_proportion": np.float64(0.25),
        "flag": np.bool_(True),
    }
    text = json.dumps(payload, cls=HubJsonEncoder)
    out = json.loads(text)
    assert out == {"nan_count": 3, "nan_proportion": 0.25, "flag": True}


def test_dataset_meta_statistics_round_trip():
    from muller.core.meta.dataset_meta import DatasetMeta

    meta = DatasetMeta()
    meta.statistics = {
        "num_examples": 10,
        "statistics": [
            {
                "column_name": "x",
                "column_type": "float",
                "column_statistics": {
                    "nan_count": np.int64(0),
                    "nan_proportion": np.float64(0.0),
                },
            }
        ],
    }
    # nbytes -> tobytes() uses HubJsonEncoder
    assert meta.nbytes > 0
    raw = meta.tobytes()
    loaded = DatasetMeta.frombuffer(raw)
    assert loaded.statistics["num_examples"] == 10
    assert loaded.statistics["statistics"][0]["column_statistics"]["nan_count"] == 0

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin


def write_to_parquet(dataset, path, columns=None):
    """Write dataset to parquet format.

    Args:
        dataset: The dataset to export.
        path: The path where to save the parquet file.
        columns: Optional list of columns to export.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_dataset = dataset.to_arrow()
    arrow_table = arrow_dataset.to_table(columns)
    writer = pa.BufferOutputStream()
    pq.write_table(arrow_table, writer)
    dataset.storage[path] = bytes(writer.getvalue())

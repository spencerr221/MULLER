# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import muller


def add_data(
        ds,
        org_dicts=None,
        schema=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True
):
    """Add data to the dataset. (called by self.add_data_from_file() and self.add_data_from_dataframes())"""
    keys = list(ds.tensors)

    if not schema:
        schema = list(org_dicts[0].keys())
    else:
        schema = muller.api.dataset_api.DatasetAPI.convert_schema(schema)

    if not all(col in keys for col in schema):
        raise ValueError("The column names in schema do not match the dataset keys.")

    @muller.compute
    def data_to_muller(data, sample_out):
        for col_name in schema:
            sample_out[col_name].append(data[col_name])
        return sample_out

    if workers in (0, 1):
        with ds:
            for data in org_dicts:
                for col in schema:
                    ds[col].append(data[col])

    else:
        with ds:
            data_to_muller().eval(org_dicts, ds, num_workers=workers,
                                scheduler=scheduler, disable_rechunk=disable_rechunk,
                                progressbar=progressbar, ignore_errors=ignore_errors)

    return ds

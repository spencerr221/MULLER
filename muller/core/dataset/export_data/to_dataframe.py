# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import pandas as pd


def to_dataframe(dataset, tensor_list=None, index_list=None):
    """ Export dataset as a pandas dataframe. """
    data = {}
    if tensor_list:
        target_list = tensor_list
    else:
        target_list = list(dataset.tensors)
    if index_list:
        for tensor in target_list:
            data.update({tensor: list(dataset[index_list][tensor].numpy().flatten())})  # 这里可以用numpy_continuous?
    else:
        for tensor in target_list:
            data.update({tensor: list(dataset[tensor].numpy().flatten())})  # 这里可以用numpy_full? 但有进程lock风险
    df = pd.DataFrame(data, columns=target_list)
    return df

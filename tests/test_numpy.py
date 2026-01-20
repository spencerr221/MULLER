# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import random

import numpy as np
import pytest

import muller
from tests.constants import TEST_NUMPY_PATH, SAMPLE_FILES
from tests.utils import official_path, official_creds


def get_raw_data():
    return [
        ['1．使用if-else语句，实现比较功能', 'if in1 > in2:', '{"通顺性评分": 5.0, "完整性评分": 3.0}]}', '综合质量评分:6分', 6.0, 5.0],
        ['编写一个Python程序，使用正则表达式匹配手机号码。', '首先，我们需要导入re模块。', '{"通顺性评分": 5.0, "完整性评分": 2.0}]}', '综合质量评分:0分', 0.0, 4.0],
        ['用C语言编译：计算圆柱的表面积与体积', '定义了四个变量:`r`,`h`,`表面积`,`体积`', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:6分', 6.0, 6.0],
        ['怎么将rgb三个通道的值合成一个rgb值', 'Java中，将提取出的R、G、B通道的值相加来合成一个RGB值。', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:7分', 7.0, 4.0],
        ['给我生成一个游戏昵称', '风之旅者', '{"通顺性评分": 5.0, "完整性评分": 5.0}]}', '综合质量评分:5分', 5.0, 7.0],
        ['写一段基于深度学习的智能客服系统设计', '基于深度学习的智能客服系统，包括数据预处理、模型构建、模型训练和预测等步骤。', '{"通顺性评分": 5.0, "完整性评分": 5.0}]}', '综合质量评分:6分', 6.0, 3.0]]


def create_dataset(storage):
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)
    tensors = ["ori_query", "ori_response", "query_analysis", "result", "score", "type"]
    ds.create_tensor("ori_query", htype="text", exist_ok=True)
    ds.create_tensor("ori_response", htype="text", exist_ok=True)
    ds.create_tensor("query_analysis", htype="text", exist_ok=True)
    ds.create_tensor("result", htype="text", exist_ok=True)
    ds.create_tensor("score", htype="generic", exist_ok=True, dtype="float64")
    ds.create_tensor("type", htype="generic", exist_ok=True, dtype="float64")
    np_data = np.array(get_raw_data())
    for i, item in enumerate(tensors):
        ds[item].extend(np_data[:, i].astype(ds[item].dtype))
    return ds


def create_dataset_with_fixed_shape(storage):
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="age", dtype="int", max_chunk_size=7)
    for i in range(20):
        ds.age.append([i])
    ds.create_tensor(name="height", dtype="float64", chunk_compression="lz4")
    for i in range(150, 170):
        ds.height.append([i])
    ds.create_tensor(name="photo", htype="image", sample_compression="jpg")
    ds.photo.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    return ds


def create_text_dataset(storage):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "[unused10][unused9]助手",
              "我是deepseek，迅雷不及掩耳盗铃儿响叮当仁不让世界充满爱之势!你是谁？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)

    with ds:
        ds.create_tensor('value', htype='text')
        ds.value.extend(values*2000)
    return ds


def test_read_single_sample(storage):
    ds = create_dataset(storage)

    assert len(ds[3].numpy(aslist=True, asrow=True)) == 1
    assert ds[3].numpy(aslist=True, asrow=True)[0]["ori_query"] == "怎么将rgb三个通道的值合成一个rgb值"

    assert len(ds[3].numpy(aslist=True, asrow=False)) == 6
    assert ds[3].numpy(aslist=True, asrow=True)[0] == ds[3].numpy(aslist=True, asrow=False)


def test_read_dataset_slices(storage):
    ds = create_dataset(storage)

    assert len(ds[3:6].numpy(aslist=True, asrow=True)) == 3
    assert ds[3:6].numpy(aslist=True, asrow=True)[1]["ori_query"] == "给我生成一个游戏昵称"

    assert len(ds[3:6].numpy(aslist=True, asrow=False)) == 6
    assert len(ds[3:6].numpy(aslist=True, asrow=False)["ori_response"]) == 3
    assert ds[3:6].numpy(aslist=True, asrow=False)["result"][2] == "综合质量评分:6分"


def test_read_dataset_slices_with_step(storage):
    ds = create_dataset(storage)

    assert len(ds[3:6:2].numpy(aslist=True, asrow=True)) == 2
    assert ds[3:6:2].numpy(aslist=True, asrow=True)[1]["ori_query"] == "写一段基于深度学习的智能客服系统设计"

    assert len(ds[3:6:2].numpy(aslist=True, asrow=False)) == 6
    assert len(ds[3:6:2].numpy(aslist=True, asrow=False)["ori_response"]) == 2
    assert ds[3:6:2].numpy(aslist=True, asrow=False)["result"][0] == "综合质量评分:7分"


def test_tensors_with_different_length(storage):
    ds = create_dataset(storage)

    ds.ori_query.append("阅读这个页面的文本")
    ds.ori_response.append("当然，我可以帮助您阅读文本。您可以将页面的链接或者截图发给我，或者您可以直接在这个对话框中粘贴文本。请告诉我您需要我阅读的页面。")
    # without query_analysis
    ds.result.append("综合质量评分:7分")
    ds.score.append(7.0)
    ds.type.append(5.0)

    assert len(ds[5:].numpy(aslist=True, asrow=False)["result"]) == 2
    assert len(ds[5:].numpy(aslist=True, asrow=False)["query_analysis"]) == 1
    assert ds[5:].numpy(aslist=True, asrow=False)["ori_query"][0] == "写一段基于深度学习的智能客服系统设计"
    assert ds[5:].numpy(aslist=True, asrow=False)["ori_response"][1] == "当然，我可以帮助您阅读文本。您可以将页面的链接或者截图发给我，或者您可以直接在这个对话框中粘贴文本。请告诉我您需要我阅读的页面。"

    with pytest.raises(ValueError) as e:
        ds[5:].numpy(aslist=True, asrow=True)
    assert e.type == ValueError
    print(e.value)
    assert str(e.value) == "The number of samples in each tensor is different or the number not equal to the length of dataset index. Please set asrow = False."


def test_fixed_shape(storage):
    ds = create_dataset_with_fixed_shape(storage)
    assert ds.age[4].numpy() == [4]
    assert ds.age[17].numpy() == [17]
    assert ds.height[3].numpy() == [153.]
    assert ds.height[19].numpy() == [169.]
    assert ds.photo[0][0][0][0].numpy() == 243


def test_numpy_batch_random_access(storage):
    ds = create_text_dataset(storage)

    def old_random_access(ds, index_list):
        line_list = []
        for i in index_list:
            line = ds.value[i].numpy(fetch_chunks=True)
            line_list.append(line)
        return line_list

    def new_random_access(ds, index_list):
        line_list = []
        line_list.extend(ds.value.numpy_batch_random_access(index_list=index_list, parallel="threaded"))
        return line_list

    random_list = [random.randint(0, 9999) for _ in range(10)]

    list_1 = old_random_access(ds, index_list=random_list)
    list_2 = new_random_access(ds, index_list=random_list)

    assert list_1 == list_2


if __name__ == '__main__':
    pytest.main(["-s", "test_numpy.py"])

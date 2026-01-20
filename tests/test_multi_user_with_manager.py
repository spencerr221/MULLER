# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pytest

import muller
from muller.util.exceptions import UnAuthorizationError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import TEST_MULTI_USER_PATH
from tests.utils import official_path, official_creds

def test_manager_user(storage):
    """ test the authorization of the manager user who create the dataset."""
    # 1. 用户manager
    SensitiveConfig().uid = "manager"
    # 创建数据集
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)

    ds.commit()

    # 2. 切换到用户B
    SensitiveConfig().uid = "B"
    # 查询数据，创建dev-B分支，添加数据
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage))
    assert len(ds.filter_vectorized([("labels", "==", 10)]).filtered_index) == 2
    ds.checkout("dev-B", create=True)
    ds.labels.extend([1] * 10)
    assert len(ds) == 30

    # 创建dev-B1分支，删除数据，查询数据
    ds.checkout("dev-B1", create=True)
    ds.pop([0, 1, 2])
    assert len(ds.filter_vectorized([("labels", "==", 10)]).filtered_index) == 1
    assert len(ds) == 27

    # 再创建dev-B2分支，更新数据,查询数据
    ds.checkout("dev-B2", create=True)
    ds.labels[0] = 5
    assert len(ds.filter_vectorized([("labels", "==", 4)]).filtered_index) == 1
    assert len(ds) == 27
    ds.commit()

    # 3. 切换回用户manager
    SensitiveConfig().uid = "manager"
    # 在主分支添加数据，查询数据
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage))
    ds.labels.extend([2] * 10)
    assert len(ds.filter_vectorized([("labels", "==", 2)]).filtered_index) == 12

    # 切换到dev-B分支，在dev-B分支添加数据【manager有所有分支的写权限】
    ds.checkout("dev-B")
    ds.labels.extend([3] * 10)
    assert len(ds.filter_vectorized([("labels", "==", 3)]).filtered_index) == 10
    assert len(ds) == 40

    # 切换到dev-B1分支，这个分支添加数据【manager有所有分支的写权限】
    ds.checkout("dev-B1")
    ds.labels.extend([4] * 10)
    assert len(ds.filter_vectorized([("labels", "==", 4)]).filtered_index) == 12
    assert len(ds) == 37

    # 在dev-B1分支上合并dev-B2分支【manager有所有分支的写权限】
    ds.merge("dev-B2")
    assert ds.labels[0].numpy() == [5]

    # 4. 切换回用户B
    SensitiveConfig().uid = "B"
    # 切换到dev-B分支，合并dev-B1分支
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH) + "@dev-B",
                       creds=official_creds(storage))
    assert len(ds) == 40
    ds.merge("dev-B1", append_resolution='theirs', update_resolution='theirs', pop_resolution="theirs")
    assert len(ds) == 37

    # 切换到main分支，合并dev-B分支【用户B没有分支main的写权限，这里会报UnAuthorizationError，结束！】
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage))
    assert len(ds) == 30
    try:
        ds.merge("dev-B", append_resolution='theirs', update_resolution='theirs', pop_resolution="theirs")
    except UnAuthorizationError as e:
        assert True, f"exception: {e}"


def test_unauthorization_case(storage):
    """ test catching the unauthorization error and ensure the next action is not affected. """
    # 1. 用户A
    SensitiveConfig().uid = "A"
    # 创建数据集
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.create_tensor('categories', htype='text')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    ds.categories.extend(['agent', '情感', '生成', '写作', '情感', 'agent', '生成', '写作', '情感', '写作'] * 2)
    ds.commit()

    SensitiveConfig().uid = "B"

    try:
        ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100] * 2)
    except UnAuthorizationError:
        pass

    assert len(ds.labels.numpy(aslist=True)) == 20

    try:
        ds.pop(0)
    except UnAuthorizationError:
        pass

    assert len(ds) == 20

    try:
        ds[7].update({"labels": 3})
    except UnAuthorizationError:
        pass

    assert ds[7].labels.numpy(aslist=True) == [8]

    try:
        ds.checkout("dev", create=True)
        assert True, "No exception raises"
    except Exception as e:
        assert False, f"exception: {e}"


def test_dataset_creator(storage):
    """ test the authorization of the dataset creator."""
    SensitiveConfig().uid = "A"
    ds = muller.dataset(path=official_path(storage, TEST_MULTI_USER_PATH),
                       creds=official_creds(storage), overwrite=True)
    SensitiveConfig().uid = "B"
    ds.checkout("dev-B", create=True)
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.labels.extend([10, 2, 30, 4, 50, 6, 70, 8, 90, 100])
    ds.commit()
    assert len(ds) == 10

    SensitiveConfig().uid = "A"
    ds.checkout("main")
    ds.create_tensor('labels', htype='generic', dtype='int')
    ds.labels.extend([10, 2, 30])
    ds.commit()
    assert len(ds) == 3
    ds.checkout("dev-B")
    assert len(ds) == 10
    ds.labels.extend([10, 2])
    ds.commit()
    assert len(ds) == 12

    SensitiveConfig().uid = "B"
    ds.checkout("main")
    with pytest.raises(UnAuthorizationError):
        ds.labels.extend([10, 2, 30])

    ds = muller.load(path=official_path(storage, TEST_MULTI_USER_PATH), creds=official_creds(storage))
    SensitiveConfig().uid = "A"
    ds.merge("dev-B", append_resolution='theirs', update_resolution='theirs', pop_resolution="theirs")
    assert len(ds) == 12


if __name__ == '__main__':
    pytest.main(["-s", "test_multi_user_with_manager.py"])

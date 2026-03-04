# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Metadata population utilities for Dataset."""

from typing import Optional

from muller.core.auth.authorization import obtain_current_user
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.storage_keys import dataset_exists, get_dataset_diff_key, get_dataset_meta_key
from muller.core.version_control.dataset_diff import DatasetDiff
from muller.core.version_control.functions import load_meta
from muller.util.exceptions import (
    CouldNotCreateNewDatasetException,
    PathNotEmptyException,
    VersionControlError,
)


def populate_meta(dataset, address: Optional[str] = None, verbose=True):
    """Populates the meta information for the dataset."""
    if address is None:
        commit_id = dataset._get_commit_id_for_address("main", dataset.version_state)
    else:
        commit_id = dataset._get_commit_id_for_address(address, dataset.version_state)

    if dataset_exists(dataset.storage, commit_id):
        load_meta(dataset)
    elif not dataset.storage.empty():
        raise PathNotEmptyException
    else:
        if dataset.read_only:
            raise CouldNotCreateNewDatasetException(dataset.path)
        try:
            commit_id = dataset.version_state["commit_id"]
        except KeyError as e:
            raise VersionControlError from e

        meta = DatasetMeta()
        meta.set_dataset_creator(obtain_current_user())
        key = get_dataset_meta_key(commit_id)
        dataset.version_state["meta"] = meta
        dataset.storage.register_muller_object(key, meta)

        dataset_diff = DatasetDiff()
        key = get_dataset_diff_key(commit_id)
        dataset.storage.register_muller_object(key, dataset_diff)

        dataset.flush()

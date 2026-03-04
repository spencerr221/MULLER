# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Access control utilities for Dataset."""

from typing import Optional

import muller.constants
from muller.core.lock import lock_dataset, unlock_dataset
from muller.core.storage.local import LocalProvider
from muller.core.storage.lru_cache import LRUCache
from muller.util.exceptions import LockedException, ReadOnlyModeError


_LOCKABLE_STORAGES = {LocalProvider}


def set_read_only(dataset, value: bool, err: bool):
    """Set the read only variable."""
    storage = dataset.storage
    dataset.__dict__["_read_only"] = value

    if value:
        storage.enable_readonly()
        if isinstance(storage, LRUCache) and storage.next_storage is not None:
            storage.next_storage.enable_readonly()
        unlock_dataset(dataset)
    else:
        try:
            locked = lock(dataset, err=err)
            if locked:
                dataset.storage.disable_readonly()
                if isinstance(storage, LRUCache) and storage.next_storage is not None:
                    storage.next_storage.disable_readonly()
            else:
                dataset.__dict__["_read_only"] = True
        except LockedException as e:
            dataset.__dict__["_read_only"] = True
            if err:
                raise e


def lock(dataset, err=False, verbose=True):
    """Lock the dataset."""
    if not dataset.is_head_node or not dataset._locking_enabled:
        return True
    storage = dataset.base_storage
    if storage.read_only and not dataset._locked_out:
        if err:
            raise ReadOnlyModeError()
        return False

    if isinstance(storage, tuple(_LOCKABLE_STORAGES)) and (
            not dataset.read_only or dataset._locked_out
    ):
        if not muller.constants.LOCK_LOCAL_DATASETS and isinstance(storage, LocalProvider):
            return True
        try:
            storage.disable_readonly()
            lock_dataset(dataset, lock_lost_callback=dataset._lock_lost_handler)
        except LockedException as e:
            set_read_only(dataset, True, False)
            dataset.__dict__["_locked_out"] = True
            if err:
                raise e
            return False
    return True


def enable_admin_mode(dataset, password: Optional[str] = None):
    """Enable admin mode for dataset creator to modify all branches."""
    from muller.core.auth.authorization import obtain_current_user
    from muller.util.exceptions import UnAuthorizationError
    from muller.client.log import logger

    current_user = obtain_current_user()

    try:
        creator = dataset.version_state.get("meta").dataset_creator if dataset.version_state.get("meta") else None
    except (TypeError, AttributeError):
        try:
            creator = dataset.obtain_dataset_creator_name_from_storage()
        except Exception:
            creator = None

    if not creator or current_user != creator:
        raise UnAuthorizationError(
            f"Only dataset creator [{creator}] can enable admin mode. Current user: [{current_user}]"
        )

    dataset._admin_mode = True
    logger.info(f"Admin mode enabled for user [{current_user}]")


def disable_admin_mode(dataset):
    """Disable admin mode."""
    from muller.client.log import logger

    dataset._admin_mode = False
    logger.info("Admin mode disabled")

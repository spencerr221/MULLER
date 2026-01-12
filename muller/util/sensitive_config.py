# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from functools import wraps


def Singleton(cls):
    """
    This function is intended to implement a decorator—specifically a type decorator.
    cls: the name of the singleton class being defined.
    """
    instance = {}

    @wraps(cls)
    def singleton(*args, **kargs):
        if cls not in instance:
            # If the class cls does not exist, create it,
            # and store the instance created by this class in a dictionary.
            instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return singleton


@Singleton
class SensitiveConfig:
    """
    The singleton pattern is a design pattern that ensures a class has only one instance
    throughout the lifetime of a program.
    """

    def __init__(self):
        self._uid = "public"

    @property
    def uid(self):
        """set uid"""
        return self._uid

    @uid.setter
    def uid(self, uid_value):
        self._uid = uid_value

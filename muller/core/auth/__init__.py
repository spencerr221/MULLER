# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core authentication and authorization module for MULLER."""

from muller.core.auth.authorization import obtain_current_user

__all__ = [
    'obtain_current_user',
]

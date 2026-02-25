# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.auth.permission.invalid_user_op

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.auth.permission.invalid_user_op instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.auth.permission.invalid_user_op import *  # noqa: F401, F403

warnings.warn(
    "muller.util.permission.invalid_user_op is deprecated and will be removed in a future version. "
    "Please use muller.core.auth.permission.invalid_user_op instead.",
    DeprecationWarning,
    stacklevel=2
)

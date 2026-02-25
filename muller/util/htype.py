# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Deprecated: This module has been moved to muller.core.types.htype

This module is kept for backward compatibility and will be removed in a future version.
Please update your imports to use muller.core.types.htype instead.
"""

import warnings

# Re-export all functions from the new location
from muller.core.types.htype import *  # noqa: F401, F403

warnings.warn(
    "muller.util.htype is deprecated and will be removed in a future version. "
    "Please use muller.core.types.htype instead.",
    DeprecationWarning,
    stacklevel=2
)

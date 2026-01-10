# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/compression.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import itertools
import re
import struct
from typing import Optional

from PIL import Image

# ---- The compression methods for different type of files, including byte, image, video, audio, etc.
BYTE_COMPRESSIONS = [
    "lz4"
]

IMAGE_COMPRESSIONS = [
    "bmp", "dib", "eps", "fli", "gif", "ico", "im",
    "jpg", "jpeg", "jpeg2000",
    "msp", "mpo", "pcx", "png", "ppm", "sgi",
    "tga", "tiff", "webp", "wmf", "xbm"
]

IMAGE_COMPRESSION_EXT_DICT = {
    # "apng": [".png"],
    "bmp": [".bmp"],
    "eps": [".eps"],
    "fli": [".fli"],
    "dib": [".dib"],
    "gif": [".gif"],
    "ico": [".ico"],
    "im": [".im"],
    "jpeg": [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp"],
    "jpeg2000": [
        ".jp2",
        ".j2k",
        ".jpf",
        ".jpm",
        ".jpg2",
        ".j2c",
        ".jpc",
        ".jpx",
        ".mj2",
    ],
    "msp": [".msp"],
    "mpo": [".mpo"],
    "pcx": [".pcx"],
    "png": [".png"],
    "ppm": [".pbm", ".pgm", ".ppm", ".pnm"],
    "sgi": [".sgi"],
    "tga": [".tga"],
    "tiff": [".tiff", ".tif"],
    "webp": [".webp"],
    "wmf": [".wmf"],
    "xbm": [".xbm"],
}

COMPRESSION_ALIASES = {"jpg": "jpeg", "tif": "tiff", "jp2": "jpeg2000"}
IMAGE_COMPRESSION_EXTENSIONS = list(
    set(itertools.chain(*IMAGE_COMPRESSION_EXT_DICT.values()))
)

# Pillow plugins for some formats might not be installed:
Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]

# IMAGE_COMPRESSIONS.insert(0, "apng")
IMAGE_COMPRESSIONS.insert(1, "dcm")
IMAGE_COMPRESSIONS.insert(2, "mpo")
IMAGE_COMPRESSIONS.insert(3, "fli")

VIDEO_COMPRESSIONS = ["mp4", "mkv", "avi"]
AUDIO_COMPRESSIONS = ["mp3", "flac", "wav"]
NIFTI_COMPRESSIONS = ["nii", "nii.gz"]

READONLY_COMPRESSIONS = [
    "mpo", "fli", "dcm",
    *NIFTI_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS
]

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS,
    *NIFTI_COMPRESSIONS,
]

SUPPORTED_COMPRESSIONS = list(sorted(set(SUPPORTED_COMPRESSIONS)))  # type: ignore
SUPPORTED_COMPRESSIONS.append(None)  # type: ignore

# ------- Determine the file types based on the given compression/extension name.

_compression_types = {}
for c in IMAGE_COMPRESSIONS:
    _compression_types[c] = "image"
for c in BYTE_COMPRESSIONS:
    _compression_types[c] = "byte"
for c in VIDEO_COMPRESSIONS:
    _compression_types[c] = "video"
for c in AUDIO_COMPRESSIONS:
    _compression_types[c] = "audio"
for c in NIFTI_COMPRESSIONS:
    _compression_types[c] = "nifti"

# ----JPEG markers
_JPEG_SOFS = [
    b"\xff\xc0",
    b"\xff\xc2",
    b"\xff\xc1",
    b"\xff\xc3",
    b"\xff\xc5",
    b"\xff\xc6",
    b"\xff\xc7",
    b"\xff\xc9",
    b"\xff\xca",
    b"\xff\xcb",
    b"\xff\xcd",
    b"\xff\xce",
    b"\xff\xcf",
    b"\xff\xde",
    # Skip:
    b"\xff\xcc",
    b"\xff\xdc",
    b"\xff\xdd",
    b"\xff\xdf",
    # App: (0xFFE0 - 0xFFEF):
    *map(lambda x: x.to_bytes(2, "big"), range(0xFFE0, 0xFFF0)),
    # DQT:
    b"\xff\xdb",
    # COM:
    b"\xff\xfe",
    # Start of scan
    b"\xff\xda",
]

_JPEG_SKIP_MARKERS = set(_JPEG_SOFS[14:])
_JPEG_SOFS_RE = re.compile(b"|".join(_JPEG_SOFS))
_STRUCT_HHB = struct.Struct(">HHB")
_STRUCT_II = struct.Struct(">ii")

# Note: The bellow codes can be optimized.
BYTE_COMPRESSION = "byte"
IMAGE_COMPRESSION = "image"
VIDEO_COMPRESSION = "video"
AUDIO_COMPRESSION = "audio"
POINT_CLOUD_COMPRESSION = "point_cloud"
MESH_COMPRESSION = "mesh"
NIFTI_COMPRESSION = "nifti"

COMPRESSION_TYPES = [
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    AUDIO_COMPRESSION,
    VIDEO_COMPRESSION,
    POINT_CLOUD_COMPRESSION,
    MESH_COMPRESSION,
    NIFTI_COMPRESSION,
]


def get_compression_type(c):
    """Returns the compression type for the given compression name."""
    if c is None:
        return None
    ret = _compression_types.get(c)
    if ret is None and c.upper() in Image.OPEN:
        ret = IMAGE_COMPRESSION
    if ret is None:
        raise KeyError(c)
    return ret


compression_ratios = {None: 1.0, "jpeg": 0.5, "png": 0.5, "webp": 0.5, "lz4": 0.5}


def get_compression_ratio(compression: Optional[str]) -> float:
    return compression_ratios.get(compression, 0.5)

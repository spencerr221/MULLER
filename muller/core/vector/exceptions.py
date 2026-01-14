# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from pathlib import Path


class IndexMetaError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IndexAddDataError(Exception):
    def __init__(self, message):
        super().__init__(message)


class IndexTypeNotSupportError(Exception):
    def __init__(self, index_type: str):
        super().__init__(f"index type: '{index_type}' is not support yet.")


class VectorIndexImportError(Exception):
    def __init__(self, message):
        super().__init__(message)


class FaissIndexError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IndexNotFoundError(Exception):
    def __init__(self, tensor_name: str, index_name: str):
        super().__init__(
            f"index '{index_name}' is not found in tensor '{tensor_name}'."
        )


class IndexExistsError(Exception):
    def __init__(self, tensor_name: str, index_name: str):
        super().__init__(f"index '{index_name}' is exists in tensor '{tensor_name}'.")


class CreateIndexError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class UnsupportedTypeError(Exception):
    def __init__(self, htype: str, dtype: object):
        super().__init__(
            f"Unsupported tensor type (htype:{htype}, dtype:{dtype}) to create vector index."
        )


class GetIndexError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class LoadingIndexError(Exception):
    def __init__(self, message):
        super().__init__("Error occur when loading index.")
        super().__init__(message)


class IndexFileNotFoundError(LoadingIndexError):
    def __init__(self, path: Path, index_name: str):
        super().__init__(
            f"File not found in '{path}' when loading index '{index_name}'"
        )


class SearchError(Exception):
    def __init__(self, message):
        super().__init__(message)


class IndexNotLoadError(Exception):
    def __init__(self, tensor_name: str, index_name: str):
        super().__init__(f"index {index_name} of tensor {tensor_name} is not load yet.")

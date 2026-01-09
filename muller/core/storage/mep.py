# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Set, Dict, Any

import requests

sys.path.append("/tmp/user/")

from muller.core.storage.provider import StorageProvider
from muller.util.exceptions import (DirectoryAtPathException,
                                   FileAtPathException)
from muller.util.mep.nsp import nsputil


class MepProvider(StorageProvider):
    """Provider class for using the huashan platform."""

    def __init__(self, root: str, bucket_id: str, tenant: str, mulleriles: str, nsp_host: str, nsp_id: str,
                 nsp_secret: str):

        if os.path.isfile(root):
            raise FileAtPathException(root)

        if root.endswith("/"):
            self.root = root[:-1]
        else:
            self.root = root
        self.bucket_id = bucket_id  # "nsp-semtp-datafs-p01-drcn"
        self.tenant = tenant  # "system"
        self.nsp_host = nsp_host
        self.nsp_id = nsp_id
        self.nsp_secret = nsp_secret

        self.mulleriles = mulleriles
        self.files: Optional[Set[str]] = set(eval(mulleriles))

        self.expiration: Optional[str] = None
        self.db_engine: bool = False
        self.repository: Optional[str] = None

    def __getitem__(self, key: str):
        try:
            full_path = self._check_is_file(key)
            value = self.mepReadFromObs(full_path)
            return value
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def __setstate__(self, state_tuple):
        self.__init__(state_tuple[0], state_tuple[1], state_tuple[2], state_tuple[3], state_tuple[4], state_tuple[5],
                      state_tuple[6])

    def __getstate__(self):
        return self.root, self.bucket_id, self.tenant, self.mulleriles, self.nsp_host, self.nsp_id, self.nsp_secret

    def __setitem__(self, key: str, value: bytes):
        pass

    def __delitem__(self, key: str):
        """Delete the object present at the path.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> del local_provider["abc.txt"]

        Args:
            key (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        try:
            if self.files is not None:
                self.files.discard(key)
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError from e

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> for my_data in local_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.files

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> len(local_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.files)

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(os.path.join(self.root, path))
        sd.read_only = read_only
        return sd

    def mepReadFromObs(self, object_id):
        objectId = object_id
        resJson = nsputil.getDownloadUrl("/" + objectId, self.nsp_host, self.nsp_id, self.nsp_secret)
        downloadUrl = resJson['url']
        response = requests.get(downloadUrl)
        return response.content

    def get_bytes(self,
                  path: str,
                  start_byte: Optional[int] = None,
                  end_byte: Optional[int] = None):
        try:
            full_path = self._check_is_file(path)
            value = self.mepReadFromObs(full_path)
            if start_byte is not None:
                value = value[start_byte:-1]
            if end_byte is None:
                return value
            return value[0:end_byte]
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError(path) from e

    def check_readonly(self):
        pass

    def get_object_size(self, key: str) -> str:
        pass

    def clear(self, prefix=""):
        pass

    def get_items(self, keys: Set[str], ignore_key_error: bool = False, max_workers: Optional[int] = None):
        def get_file(path: str):
            return path, self[path]

        dic = {}

        if not ignore_key_error:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(get_file, keys):
                    dic[result[0]] = result[1]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(get_file, path) for path in keys]
                for future in futures:
                    try:
                        res = future.result()
                        dic[res[0]] = res[1]
                    except KeyError:
                        pass
        return dic

    def set_items(self, contents: Dict[str, Any]):
        raise NotImplementedError(f"set_items is not implemented for {self.__class__.__name__}")

    def del_items(self, keys: Set[str]):
        raise NotImplementedError(f"del_items is not implemented for {self.__class__.__name__}")

    def _all_keys(self, refresh: bool = False) -> Set[str]:
        pass

    def _check_is_file(self, key: str):
        """The function to checks if the path is a file. Returns the full_path to file if True.

        Args:
            key (str): the path to the object relative to the root of the provider.

        Returns:
            str: the full path to the requested file.

        Raises:
            DirectoryAtPathException: If a directory is found at the path.
        """
        return f"{self.root}/{key}"

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import ctypes
import ctypes.util
import gc
import warnings

from multiprocess import Manager as _Manager
from multiprocess.pool import Pool as ProcessPool

from muller.core.compute.provider import ComputeProvider


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)
        self.pool = ProcessPool(processes=workers)
        self.manager = _Manager()
        self._closed = False

    def __del__(self):
        if not self._closed:
            self.close()
            warnings.warn(
                "process pool thread leak. check compute provider is closed after use"
            )

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()
        if self.manager:
            self.manager.shutdown()
            self.manager = None
        self.shrink_heap()
        self._closed = True

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def shrink_heap(self):
        """Function to shrink heap."""
        gc.collect()
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)

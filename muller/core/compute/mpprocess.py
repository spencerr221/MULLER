# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import multiprocessing as mp
from multiprocessing import Manager

from muller.core.compute.provider import ComputeProvider


class MPProcessProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)
        self.workers = workers
        self.pool = mp.Pool(processes=workers)
        self.manager = Manager()
        self._closed = False

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.terminate()
        self.pool.join()
        if self.manager:
            self.manager.shutdown()
            self.manager = None
        self._closed = True

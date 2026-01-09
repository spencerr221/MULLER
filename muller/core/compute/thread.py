# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from multiprocessing import Manager

from pathos.pools import ThreadPool  # type: ignore

from muller.core.compute.provider import ComputeProvider


class ThreadProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)
        self.manager = Manager()
        self.pool = ThreadPool(nodes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()

# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
from abc import abstractmethod
from typing import Union, Tuple

try:
    import faiss
except ImportError:
    print("faiss not found")
import numpy as np
from numpy._typing import NDArray
from torch import Tensor

import muller.core.vector.index_param as IndexParam
from muller.core.vector.exceptions import FaissIndexError
from muller.core.vector.utils import metric_map


class FaissVectorIndex:
    @classmethod
    @abstractmethod
    def search(
        cls,
        index: faiss.Index,
        query_vector: Union[NDArray[np.float32], Tensor],
        metric: str,
        **search_param,
    ):
        """
        Abstract search method for Faiss vector index.
        Args:
            index: the Faiss index object to call search
            query_vector: the 1-d ndarray to do knn query
            metric: the metric to measure distance between vectors
            **search_param: other search parameter for specific index type
        """
        pass

    @classmethod
    @abstractmethod
    def create(
        cls,
        vector_array: Union[NDArray[np.float32], Tensor],
        id_array: Union[NDArray[np.float32], Tensor],
        dimension: int,
        metric: str,
        **index_param,
    ):
        """
        Abstract create method for Faiss vector index.
        Args:
            vector_array: the 1-d vector array to create vector index.
            id_array: the 1-d ndarray of vectors.
            dimension: the dimension of each vector.
            metric: the metric to measure the distance between vectors.
            **index_param:
        """
        pass

    @classmethod
    def load(cls, **kwargs) -> faiss.Index:
        """
        Default method for loading Faiss vector index.
        Args:
            **kwargs: the extra load vector index parameters.

        Returns:

        """
        param = IndexParam.LoadFaissIndex.model_validate(kwargs)
        index_file = param.path / f"{param.index_name}.index"
        if index_file.exists() and index_file.is_file():
            index = faiss.read_index(str(index_file))
            if param.device == "gpu":
                index = faiss.index_cpu_to_gpu(index)
            return index
        raise IndexError(f"index file: {index_file} not found.")

    @classmethod
    def save(cls, index: faiss.Index, **kwargs):
        """
        Default method for saving Faiss vector index.
        Args:
            index: the Faiss vector index object to save.
            **kwargs: Extra parameters for saving index.
        """
        param = IndexParam.SaveFaissIndex.model_validate(kwargs)
        index_file = param.path / f"{param.index_name}.index"
        faiss.write_index(index, str(index_file))

    @classmethod
    def _get_faiss_metric(cls, metric: str):
        try:
            return metric_map[metric]
        except KeyError as err:
            logging.error(f"Get invalid metric: {metric}")
            raise FaissIndexError(f"'{metric}' not found in metric_map") from err


class IVFPQIndex(FaissVectorIndex):

    @classmethod
    def create(
        cls,
        vector_array: Union[NDArray[np.float32], Tensor],
        id_array: Union[NDArray[np.uint64], Tensor],
        dimension: int,
        metric: str,
        **index_param: object,
    ):
        # verify parameter
        param = IndexParam.CreateIVFPQ.model_validate(index_param)

        if metric == "cosine":
            faiss.normalize_L2(vector_array)
        try:
            quantizer = faiss.IndexFlat(
                dimension, FaissVectorIndex._get_faiss_metric(metric)
            )
            vector_index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                param.nlist,
                param.m,
                8,
                FaissVectorIndex._get_faiss_metric(metric),
            )
            vector_index.train(x=vector_array)
            vector_index.add(vector_array)
        except KeyError:
            logging.error(f"Get invalid metric: {metric}")
        return vector_index, param.__dict__

    @classmethod
    def search(
        cls,
        index: faiss.Index,
        query_vector: Union[NDArray[np.float32], Tensor],
        metric: str,
        **search_param,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        # verify parameter
        param = IndexParam.SearchIVFPQ.model_validate(search_param)
        if metric == "cosine":
            faiss.normalize_L2(query_vector)
        index.nprobe = param.nprobe
        index.k_factor = param.refine_factor
        dist_list: NDArray[np.float32]
        id_list: NDArray[np.int64]
        dist_list, id_list = index.search(x=query_vector, k=param.topk)
        return dist_list, id_list


class FlatIndex(FaissVectorIndex):

    @classmethod
    def create(
        cls,
        vector_array: Union[NDArray[np.float32], Tensor],
        id_array: Union[NDArray[np.uint64], Tensor],
        dimension: int,
        metric: str,
        **index_param,
    ):
        if metric == "cosine":
            faiss.normalize_L2(vector_array)
        try:
            vector_index = faiss.IndexFlat(
                dimension, FaissVectorIndex._get_faiss_metric(metric)
            )
            vector_index.add(vector_array)
        except KeyError:
            logging.error(f"get invalid metric")
        return vector_index, {}

    @classmethod
    def search(
        cls,
        index: faiss.Index,
        query_vector: Union[NDArray[np.float32], Tensor],
        metric: str,
        **search_param,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        # verify parameter
        param = IndexParam.SearchFLAT.model_validate(search_param)
        if metric == "cosine":
            faiss.normalize_L2(query_vector)
        dist_list: NDArray[np.float32]
        id_list: NDArray[np.int64]
        dist_list, id_list = index.search(x=query_vector, k=param.topk)
        return dist_list, id_list


class HNSWFLATIndex(FaissVectorIndex):

    @classmethod
    def create(
        cls,
        vector_array: Union[NDArray[np.float32], Tensor],
        id_array: Union[NDArray[np.float32], Tensor],
        dimension: int,
        metric: str,
        **index_param,
    ):
        # verify parameter
        param = IndexParam.CreateHNSW.model_validate(index_param)
        if metric == "cosine":
            faiss.normalize_L2(vector_array)

        vector_index = faiss.IndexHNSWFlat(
            dimension, param.m, FaissVectorIndex._get_faiss_metric(metric)
        )
        vector_index.hnsw.efConstruction = param.ef_construction
        vector_index.train(x=vector_array)
        vector_index.add(vector_array)

        return vector_index, param.__dict__

    @classmethod
    def search(
        cls,
        index: faiss.IndexIDMap,
        query_vector: Union[NDArray[np.float32], Tensor],
        metric: str,
        **search_param,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        # verify parameter
        param = IndexParam.SearchHNSW.model_validate(search_param)
        if metric == "cosine":
            faiss.normalize_L2(query_vector)
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", param.ef_search)
        dist_list: NDArray[np.float32]
        id_list: NDArray[np.int64]
        dist_list, id_list = index.search(query_vector, param.topk)
        return dist_list, id_list

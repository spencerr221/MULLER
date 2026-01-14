# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pathlib

from pydantic import BaseModel


class CreateIVFPQ(BaseModel):
    nlist: int = 128
    m: int = 1


class SearchIVFPQ(BaseModel):
    refine_factor: float = 1.0
    nprobe: int = 8
    topk: int = 1


class SearchFLAT(BaseModel):
    topk: int = 1


class CreateHNSW(BaseModel):
    m: int = 32
    ef_construction: int = 40


class SearchHNSW(BaseModel):
    ef_search: int = 16
    topk: int = 1


class SearchDISKANN(BaseModel):
    """Search Disk-ann index parameter class

    Parameters
    ----------
    topk: int

    complexity: int
        Size of distance ordered list of candidate neighbors to use while searching. List size increases accuracy at the
         cost of latency. Must be at least k_neighbors in size.
    beam_width: int
        The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per
        iteration of search code. Larger beamwidth will result in fewer IO round-trips per query, but might result in
        slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD
        IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search. Specifying 0 will optimize the
        beamwidth depending on the number of threads performing search, but will involve some tuning overhead.
    """

    topk: int = 1
    complexity: int = 8
    beam_width: int = 1
    num_threads: int = 0


class CreateDiskANN(BaseModel):
    """Create Disk-ann index parameter class

    Parameters
    ----------
    complexity: int
        The size of the candidate nearest neighbor list to use when building the index. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for the
        same search complexity. Use a value that is at least as large as graph_degree unless you are prepared to
        compromise on quality
    graph_degree: int
        The degree of the graph index, typically between 60 and 150. A larger maximum degree will result in larger
        indices and longer indexing times, but better search quality.
    num_nodes_to_cache: int
        Number of nodes to cache in memory (> -1)
    search_memory_maximum: float
        Build index with the expectation that the search will use at most search_memory_maximum, in gb.
    build_memory_maximum: float
        Build index using at most build_memory_maximum in gb. Building processes typically require more memory, while
        search memory can be reduced.
    num_threads: int
        Number of threads to use when creating this index. 0 is used to indicate all available logical processors
        should be used.
    pq_disk_bytes: int
        Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors
        are too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing
        on SSD. This will trade off recall. You would also want this to be greater than the number of bytes used for
        the PQ compressed data stored in-memory. Default is 0.
    path: str
        The directory containing the index files. This directory must contain the following files:

            {index_prefix}_sample_data.bin
            {index_prefix}_mem.index.data
            {index_prefix}_pq_compressed.bin
            {index_prefix}_pq_pivots.bin
            {index_prefix}_sample_ids.bin
            {index_prefix}_disk.index

        It may also include the following optional files:

            ``{index_prefix}_vectors.bin``: Optional. diskannpy builder functions may create this file in the
            index_directory if the index was created from a numpy array
            ``{index_prefix}_metadata.bin``: Optional. diskannpy builder functions create this file to store metadata
            about the index, such as vector dtype, distance metric, number of vectors and vector dimensionality. If an
            index is built from the diskann cli tools, this file will not exist.
    """

    complexity: int = 5
    graph_degree: int = 5
    num_nodes_to_cache: int = 1
    search_memory_maximum: float = 0.01
    build_memory_maximum: float = 0.01
    num_threads: int = 4
    pq_disk_bytes: int = 0
    path: str


class SaveFaissIndex(BaseModel):
    path: pathlib.Path
    index_name: str


class LoadFaissIndex(BaseModel):
    path: pathlib.Path
    index_name: str
    device: str = "cpu"


class LoadDiskANN(BaseModel):
    path: pathlib.Path
    num_threads: int = 16
    num_nodes_to_cache: int = 10

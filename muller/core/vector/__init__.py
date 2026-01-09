#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

# if using faiss-gpu this should be imported
try:
    import faiss.contrib.torch_utils
except ImportError:
    print("faiss not found")

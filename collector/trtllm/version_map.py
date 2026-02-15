# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Version map for TensorRT-LLM collectors.

No fallback - unsupported versions will raise error.
"""

VERSION_MAP = {
    "attn": {
        "v1": ["0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0"],
    },
    "gemm": {
        "v1": ["0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0"],
    },
    "mla": {
        "v1": ["0.9.0", "1.0.0", "1.1.0rc1"],
        "v2": ["1.1.0rc2", "1.1.0", "1.2.0", "1.3.0"],
    },
    "mla_bmm": {
        "v1": ["0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0"],
    },
    "moe": {
        "v1": ["0.9.0", "0.10.0"],
        "v2": ["0.11.0", "0.12.0", "0.13.0"],
        "v3": ["1.0.0", "1.1.0", "1.2.0", "1.3.0"],
    },
    "mamba2": {
        "v1": ["1.1.0", "1.2.0", "1.3.0"],
    },
    "computescale": {
        "v1": ["0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0"],
    },
    "wideep_moe_compute": {
        "v1": ["1.1.0", "1.2.0", "1.3.0"],
    },
}


def get_collector_version(op: str, version: str) -> str:
    if op not in VERSION_MAP:
        raise ValueError(f"Unknown op: {op}. Supported: {list(VERSION_MAP.keys())}")
    
    for v_tag, versions in VERSION_MAP[op].items():
        if version in versions:
            return v_tag
    
    supported = list_supported_versions(op)
    raise ValueError(f"Version {version} not supported for {op}. Supported: {supported}")


def list_supported_versions(op: str) -> list[str]:
    if op not in VERSION_MAP:
        return []
    versions = []
    for v_list in VERSION_MAP[op].values():
        versions.extend(v_list)
    return sorted(set(versions))


def list_all_ops() -> list[str]:
    return list(VERSION_MAP.keys())

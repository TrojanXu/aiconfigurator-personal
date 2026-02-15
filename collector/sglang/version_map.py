# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Version map for SGLang collectors.

No fallback - unsupported versions will raise error.
"""

VERSION_MAP = {
    "attn": {
        "v1": ["0.5.5.post3", "0.5.6.post2", "0.5.8"],
    },
    "gemm": {
        "v1": ["0.5.5.post2", "0.5.6.post2", "0.5.8"],
    },
    "mla": {
        "v1": ["0.5.5.post3", "0.5.6.post2", "0.5.8"],
    },
    "mla_bmm": {
        "v1": ["0.5.5.post3", "0.5.6.post2", "0.5.8"],
    },
    "moe": {
        "v1": ["0.5.5.post3", "0.5.6.post2", "0.5.8"],
    },
    "wideep_attn": {
        "v1": ["0.5.8"],
    },
    "wideep_deepep_moe": {
        "v1": ["0.5.8"],
    },
    "wideep_mlp": {
        "v1": ["0.5.8"],
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

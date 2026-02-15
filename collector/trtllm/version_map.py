# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Version map for TensorRT-LLM collectors.

Maps framework versions to collector implementations (v1, v2, ...).
"""

# Format: {op_name: {version_tag: [supported_versions]}}
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
        "v1": ["0.9.0", "0.10.0"],                    # pre_0_20
        "v2": ["0.11.0", "0.12.0", "0.13.0"],         # pre_1_0
        "v3": ["1.0.0", "1.1.0", "1.2.0", "1.3.0"],   # latest
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

# Latest version tag for each op
LATEST_VERSION = {op: list(versions.keys())[-1] for op, versions in VERSION_MAP.items()}


def get_collector_version(op: str, version: str) -> str:
    """Get the collector version tag for a given op and framework version."""
    if op not in VERSION_MAP:
        raise ValueError(f"Unknown op: {op}. Supported: {list(VERSION_MAP.keys())}")
    
    for v_tag, versions in VERSION_MAP[op].items():
        if version in versions:
            return v_tag
    
    return LATEST_VERSION.get(op, "v1")


def list_supported_versions(op: str) -> list[str]:
    """List all supported framework versions for a given op."""
    if op not in VERSION_MAP:
        return []
    versions = []
    for v_list in VERSION_MAP[op].values():
        versions.extend(v_list)
    return sorted(set(versions))


def list_all_ops() -> list[str]:
    """List all supported operations."""
    return list(VERSION_MAP.keys())

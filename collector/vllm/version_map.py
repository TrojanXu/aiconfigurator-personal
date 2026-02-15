# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Version map for vLLM collectors.

Maps framework versions to collector implementations (v1, v2, ...).
No fallback - unsupported versions will raise error.
"""

# Format: {op_name: {version_tag: [supported_versions]}}
VERSION_MAP = {
    "attn": {
        "v1": ["0.11.0", "0.12.0", "0.14.0"],
        "v2": ["0.15.0", "0.15.1"],
    },
    "gemm": {
        "v1": ["0.11.0", "0.12.0", "0.14.0"],
        "v2": ["0.15.0", "0.15.1"],
    },
    "mla": {
        "v1": ["0.14.0"],
        "v2": ["0.15.0", "0.15.1"],
    },
    "moe": {
        "v1": ["0.11.0", "0.12.0", "0.14.0", "0.15.0", "0.15.1"],
    },
}


def get_collector_version(op: str, version: str) -> str:
    """
    Get the collector version tag for a given op and framework version.
    
    Raises:
        ValueError: If op or version is not supported
    """
    if op not in VERSION_MAP:
        raise ValueError(f"Unknown op: {op}. Supported: {list(VERSION_MAP.keys())}")
    
    op_map = VERSION_MAP[op]
    for v_tag, versions in op_map.items():
        if version in versions:
            return v_tag
    
    supported = list_supported_versions(op)
    raise ValueError(f"Version {version} not supported for {op}. Supported: {supported}")


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

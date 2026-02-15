# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Version map for vLLM collectors.

Maps framework versions to collector implementations (v1, v2, ...).

Usage:
    from collector.vllm.version_map import get_collector_version, list_supported_versions
    
    v = get_collector_version("attn", "0.15.1")  # Returns "v2"
    collector = importlib.import_module(f".collect_attn_{v}", package="collector.vllm")
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
        "v1": ["0.11.0", "0.12.0", "0.14.0"],
        "v2": ["0.15.0", "0.15.1"],
    },
    "moe": {
        "v1": ["0.11.0", "0.12.0", "0.14.0", "0.15.0", "0.15.1"],
    },
}

# Latest version tag for each op (used as fallback)
LATEST_VERSION = {
    "attn": "v2",
    "gemm": "v2",
    "mla": "v2",
    "moe": "v1",
}


def get_collector_version(op: str, version: str) -> str:
    """
    Get the collector version tag for a given op and framework version.
    
    Args:
        op: Operation name (attn, gemm, mla, moe)
        version: Framework version (e.g., "0.15.1")
    
    Returns:
        Version tag (e.g., "v1", "v2")
    
    Raises:
        ValueError: If op is not supported
    """
    if op not in VERSION_MAP:
        raise ValueError(f"Unknown op: {op}. Supported: {list(VERSION_MAP.keys())}")
    
    op_map = VERSION_MAP[op]
    
    # Find exact version match
    for v_tag, versions in op_map.items():
        if version in versions:
            return v_tag
    
    # Fallback to latest version
    # TODO: Add semantic version comparison for better fallback
    return LATEST_VERSION.get(op, "v1")


def list_supported_versions(op: str) -> list[str]:
    """
    List all supported framework versions for a given op.
    
    Args:
        op: Operation name (attn, gemm, mla, moe)
    
    Returns:
        List of supported versions
    """
    if op not in VERSION_MAP:
        return []
    
    versions = []
    for v_list in VERSION_MAP[op].values():
        versions.extend(v_list)
    return sorted(set(versions))


def list_all_ops() -> list[str]:
    """List all supported operations."""
    return list(VERSION_MAP.keys())

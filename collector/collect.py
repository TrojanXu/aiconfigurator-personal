# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Collector Entry Point.

Provides unified interface to load version-specific collectors across all frameworks.

Usage:
    from collector.collect import get_collector, run
    
    # Get collector module
    attn_collector = get_collector("vllm", "attn", "0.15.1")
    attn_collector.run_attention(...)
    
    # Or use run() directly
    run("vllm", "attn", "0.15.1", "run_attention", batch_size=1, ...)
"""

import importlib
from typing import Any

# Framework to module mapping
FRAMEWORKS = {
    "vllm": "collector.vllm",
    "sglang": "collector.sglang",
    "trtllm": "collector.trtllm",
}


def get_collector(framework: str, op: str, version: str):
    """
    Load the appropriate collector module for a given framework, op and version.
    
    Args:
        framework: Framework name (vllm, sglang, trtllm)
        op: Operation name (attn, gemm, mla, moe, ...)
        version: Framework version (e.g., "0.15.1")
    
    Returns:
        Collector module
    
    Raises:
        ValueError: If framework, op or version is not supported
    
    Example:
        >>> collector = get_collector("vllm", "attn", "0.15.1")
        >>> collector.run_attention(batch_size=1, input_len=1024, ...)
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(FRAMEWORKS.keys())}")
    
    # Import version_map from the framework
    package = FRAMEWORKS[framework]
    version_map_module = importlib.import_module(f"{package}.version_map")
    
    # Get collector version tag (raises error if not supported)
    v_tag = version_map_module.get_collector_version(op, version)
    
    # Import collector module
    module_name = f"{package}.collect_{op}_{v_tag}"
    
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {module_name}. "
            f"Make sure collector/{framework}/collect_{op}_{v_tag}.py exists."
        ) from e


def run(framework: str, op: str, version: str, func_name: str, *args, **kwargs) -> Any:
    """
    Run a collector function.
    
    Args:
        framework: Framework name (vllm, sglang, trtllm)
        op: Operation name (attn, gemm, mla, moe, ...)
        version: Framework version (e.g., "0.15.1")
        func_name: Function name in the collector module (e.g., "run_attention")
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Function result
    
    Example:
        >>> run("vllm", "attn", "0.15.1", "run_attention", batch_size=1, input_len=1024)
    """
    collector = get_collector(framework, op, version)
    
    if not hasattr(collector, func_name):
        available = [x for x in dir(collector) if not x.startswith('_')]
        raise AttributeError(
            f"Collector for {framework}/{op} v{version} has no function '{func_name}'. "
            f"Available: {available}"
        )
    
    func = getattr(collector, func_name)
    return func(*args, **kwargs)


def list_supported_versions(framework: str, op: str) -> list[str]:
    """List all supported versions for a framework/op combination."""
    if framework not in FRAMEWORKS:
        return []
    
    package = FRAMEWORKS[framework]
    version_map_module = importlib.import_module(f"{package}.version_map")
    return version_map_module.list_supported_versions(op)


def list_all_ops(framework: str) -> list[str]:
    """List all supported operations for a framework."""
    if framework not in FRAMEWORKS:
        return []
    
    package = FRAMEWORKS[framework]
    version_map_module = importlib.import_module(f"{package}.version_map")
    return version_map_module.list_all_ops()


__all__ = [
    "get_collector",
    "run",
    "list_supported_versions",
    "list_all_ops",
]

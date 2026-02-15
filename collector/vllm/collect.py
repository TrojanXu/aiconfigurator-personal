# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Collector Entry Point.

Provides unified interface to load version-specific collectors.

Usage:
    from collector.vllm.collect import get_collector, run
    
    # Get collector module
    attn_collector = get_collector("attn", "0.15.1")
    attn_collector.run_attention(...)
    
    # Or use run() directly
    run("attn", "0.15.1", "run_attention", batch_size=1, ...)
"""

import importlib
from typing import Any, Callable

from .version_map import get_collector_version, list_supported_versions, list_all_ops


def get_collector(op: str, version: str):
    """
    Load the appropriate collector module for a given op and version.
    
    Args:
        op: Operation name (attn, gemm, mla, moe)
        version: Framework version (e.g., "0.15.1")
    
    Returns:
        Collector module
    
    Example:
        >>> collector = get_collector("attn", "0.15.1")
        >>> collector.run_attention(batch_size=1, input_len=1024, ...)
    """
    v_tag = get_collector_version(op, version)
    module_name = f".collect_{op}_{v_tag}"
    
    try:
        return importlib.import_module(module_name, package="collector.vllm")
    except ImportError as e:
        raise ImportError(
            f"Failed to import {module_name}. "
            f"Make sure collector/vllm/collect_{op}_{v_tag}.py exists."
        ) from e


def run(op: str, version: str, func_name: str, *args, **kwargs) -> Any:
    """
    Run a collector function.
    
    Args:
        op: Operation name (attn, gemm, mla, moe)
        version: Framework version (e.g., "0.15.1")
        func_name: Function name in the collector module (e.g., "run_attention")
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Function result
    
    Example:
        >>> run("attn", "0.15.1", "run_attention", batch_size=1, input_len=1024)
    """
    collector = get_collector(op, version)
    
    if not hasattr(collector, func_name):
        raise AttributeError(
            f"Collector for {op} v{version} has no function '{func_name}'. "
            f"Available: {[x for x in dir(collector) if not x.startswith('_')]}"
        )
    
    func = getattr(collector, func_name)
    return func(*args, **kwargs)


# Convenience: expose version_map functions
__all__ = [
    "get_collector",
    "run",
    "get_collector_version",
    "list_supported_versions",
    "list_all_ops",
]

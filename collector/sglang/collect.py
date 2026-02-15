# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SGLang Collector Entry Point.

Provides unified interface to load version-specific collectors.
"""

import importlib
from typing import Any

from .version_map import get_collector_version, list_supported_versions, list_all_ops


def get_collector(op: str, version: str):
    """
    Load the appropriate collector module for a given op and version.
    
    Args:
        op: Operation name (attn, gemm, mla, mla_bmm, moe, wideep_*)
        version: Framework version (e.g., "0.5.8")
    
    Returns:
        Collector module
    """
    v_tag = get_collector_version(op, version)
    module_name = f".collect_{op}_{v_tag}"
    
    try:
        return importlib.import_module(module_name, package="collector.sglang")
    except ImportError as e:
        raise ImportError(
            f"Failed to import {module_name}. "
            f"Make sure collector/sglang/collect_{op}_{v_tag}.py exists."
        ) from e


def run(op: str, version: str, func_name: str, *args, **kwargs) -> Any:
    """Run a collector function."""
    collector = get_collector(op, version)
    
    if not hasattr(collector, func_name):
        raise AttributeError(
            f"Collector for {op} v{version} has no function '{func_name}'."
        )
    
    func = getattr(collector, func_name)
    return func(*args, **kwargs)


__all__ = [
    "get_collector",
    "run",
    "get_collector_version",
    "list_supported_versions",
    "list_all_ops",
]

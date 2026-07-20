#!/usr/bin/env python3
"""历史候选池实验共用的候选池改写 helper。"""

from __future__ import annotations

try:
    from ...pool_change_common import apply_pool_change
except ImportError:
    from pool_change_common import apply_pool_change

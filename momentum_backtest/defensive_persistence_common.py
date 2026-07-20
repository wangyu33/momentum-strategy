#!/usr/bin/env python3
"""围绕 defensive persistence 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_defensive_persistence import build_persistent_mask
except ImportError:
    from compare_defensive_persistence import build_persistent_mask

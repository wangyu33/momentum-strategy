#!/usr/bin/env python3
"""围绕 resource guard 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_resource_guards import RESOURCE_CODE, RESOURCE_ETF, run_variant, summarize
except ImportError:
    from compare_resource_guards import RESOURCE_CODE, RESOURCE_ETF, run_variant, summarize


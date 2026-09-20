#!/usr/bin/env python3
"""运行时环境辅助函数。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _find_workspace_dir(module_path: Path) -> Path:
    """向上寻找包含 `.tools/python_packages` 的工作区根目录。"""
    for candidate in [module_path.parent, *module_path.parents]:
        if (candidate / ".tools" / "python_packages").exists():
            return candidate
    return module_path.parents[1]


def _find_package_root(module_path: Path) -> Path:
    """向上寻找包含正式回测入口的包根目录。"""
    for candidate in [module_path.parent, *module_path.parents]:
        if (candidate / "run_backtest.py").exists() and (candidate / "runtime_env.py").exists():
            return candidate
    return module_path.parent


def prepare_local_imports(module_file: str | Path, *, include_module_dir: bool = True) -> None:
    """为脚本补齐本目录与工作区私有依赖目录，并清理用户 site-packages 污染。"""
    module_path = Path(module_file).resolve()
    module_dir = module_path.parent
    workspace_dir = _find_workspace_dir(module_path)
    package_root = _find_package_root(module_path)

    if include_module_dir and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    local_site_packages = workspace_dir / ".tools" / "python_packages"
    if local_site_packages.exists() and str(local_site_packages) not in sys.path:
        # `.tools/python_packages` 只作为兜底补充，避免覆盖当前解释器
        # 已安装且 ABI 匹配的 site-packages。
        sys.path.append(str(local_site_packages))

    scrub_user_site_packages()


def scrub_user_site_packages() -> None:
    """移除用户目录下的 site-packages，避免本地架构不匹配的依赖污染脚本运行。"""
    home = str(Path.home())
    blocked_markers = (
        f"{home}/Library/Python",
        f"{home}/.local/lib",
    )
    sys.path[:] = [
        path
        for path in sys.path
        if not path or not any(str(path).startswith(marker) for marker in blocked_markers)
    ]


def configure_matplotlib_env() -> None:
    """为 Matplotlib 指定工作区内可写缓存目录，避免导入时回退到临时目录。"""
    mplconfigdir = Path(__file__).resolve().parents[1] / ".mplconfig"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def write_json_atomic(path: Path, payload: object) -> None:
    """以原子替换方式写 JSON 文件，避免中途中断留下半截内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_text_atomic(path: Path, content: str) -> None:
    """以原子替换方式写文本文件，避免中途中断留下半截内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=path.suffix or ".txt",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

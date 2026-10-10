#!/usr/bin/env python3
"""Lightweight desktop shell for the momentum workflow."""

from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable
PYTHONPATH_VALUE = str(REPO_ROOT / ".tools" / "python_packages")
CORE_OUTPUT_DIR = REPO_ROOT / "momentum_backtest" / "output" / "core"
MONITOR_OUTPUT_DIR = REPO_ROOT / "momentum_backtest" / "output" / "monitor"


TASKS: dict[str, list[str]] = {
    "正式回测": ["momentum_backtest/run_backtest.py"],
    "正式通知补发": ["momentum_backtest/daily_monitor.py", "--force", "--debug"],
    "预览通知": ["momentum_backtest/daily_monitor.py", "--preview-send", "--debug"],
    "正式动量榜单补发": ["momentum_backtest/daily_momentum_board.py", "--force", "--debug"],
    "预览动量榜单": ["momentum_backtest/daily_momentum_board.py", "--preview-send", "--debug"],
}


def read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        last_row: dict[str, str] | None = None
        for row in reader:
            last_row = row
    return last_row


def read_selected_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            theme = str(row.get("theme", "")).strip()
            if code:
                mapping[code] = f"{name} / {theme}" if theme else name
    return mapping


def format_pct(value: str | None, digits: int = 2) -> str:
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "--"


def format_float(value: str | None, digits: int = 4) -> str:
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


class MomentumDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Momentum Strategy Desktop")
        self.root.geometry("1180x760")
        self.root.minsize(1080, 680)

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.process_lock = threading.Lock()
        self.active_process: subprocess.Popen[str] | None = None
        self.active_task_name: str | None = None

        self.summary_vars = {
            "date": tk.StringVar(value="--"),
            "nav": tk.StringVar(value="--"),
            "drawdown": tk.StringVar(value="--"),
            "signal": tk.StringVar(value="--"),
            "holding": tk.StringVar(value="--"),
            "target_exposure": tk.StringVar(value="--"),
            "momentum": tk.StringVar(value="--"),
        }
        self.run_state_var = tk.StringVar(value="空闲")

        self._configure_style()
        self._build_layout()
        self.refresh_summary()
        self.root.after(150, self._drain_log_queue)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#f3f1ea")
        style.configure("App.TFrame", background="#f3f1ea")
        style.configure("Panel.TFrame", background="#fbfaf6", relief="flat")
        style.configure("Title.TLabel", background="#f3f1ea", foreground="#1f2937", font=("Helvetica Neue", 24, "bold"))
        style.configure("SubTitle.TLabel", background="#f3f1ea", foreground="#4b5563", font=("Helvetica Neue", 12))
        style.configure("PanelTitle.TLabel", background="#fbfaf6", foreground="#111827", font=("Helvetica Neue", 14, "bold"))
        style.configure("MetricLabel.TLabel", background="#fbfaf6", foreground="#6b7280", font=("Helvetica Neue", 11))
        style.configure("MetricValue.TLabel", background="#fbfaf6", foreground="#111827", font=("Helvetica Neue", 16, "bold"))
        style.configure("RunState.TLabel", background="#f3f1ea", foreground="#9a3412", font=("Helvetica Neue", 11, "bold"))
        style.configure("Action.TButton", font=("Helvetica Neue", 12, "bold"), padding=(10, 10))
        style.configure("Link.TButton", font=("Helvetica Neue", 11), padding=(8, 6))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Momentum Strategy Desktop", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="本地桌面入口，直接驱动正式回测、通知补发和动量榜单。",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(18, 12))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)

        summary_panel = ttk.Frame(top, style="Panel.TFrame", padding=18)
        summary_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_summary_panel(summary_panel)

        action_panel = ttk.Frame(top, style="Panel.TFrame", padding=18)
        action_panel.grid(row=0, column=1, sticky="nsew")
        self._build_action_panel(action_panel)

        bottom = ttk.Frame(outer, style="App.TFrame")
        bottom.pack(fill="both", expand=True)
        bottom.columnconfigure(0, weight=5)
        bottom.columnconfigure(1, weight=2)

        log_panel = ttk.Frame(bottom, style="Panel.TFrame", padding=18)
        log_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_log_panel(log_panel)

        side_panel = ttk.Frame(bottom, style="Panel.TFrame", padding=18)
        side_panel.grid(row=0, column=1, sticky="nsew")
        self._build_side_panel(side_panel)

    def _build_summary_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="最新核心结果", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        metrics = [
            ("最新日期", "date"),
            ("净值", "nav"),
            ("当前回撤", "drawdown"),
            ("当前信号", "signal"),
            ("确认持仓", "holding"),
            ("目标仓位", "target_exposure"),
            ("25日动量", "momentum"),
        ]
        for index, (label, key) in enumerate(metrics, start=1):
            col = (index - 1) % 2
            row = ((index - 1) // 2) * 2 + 1
            frame = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 10, 10, 6))
            frame.grid(row=row, column=col, sticky="nw")
            ttk.Label(frame, text=label, style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(frame, textvariable=self.summary_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(3, 0))

        button_row = ttk.Frame(parent, style="Panel.TFrame")
        button_row.grid(row=10, column=0, columnspan=4, sticky="w", pady=(16, 0))
        ttk.Button(button_row, text="刷新摘要", style="Link.TButton", command=self.refresh_summary).pack(side="left")
        ttk.Label(button_row, textvariable=self.run_state_var, style="RunState.TLabel").pack(side="left", padx=(14, 0))

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="执行入口", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="一次只跑一个任务，日志会实时刷到下方。",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        self.action_buttons: list[ttk.Button] = []
        for name in ("正式回测", "正式通知补发", "预览通知", "正式动量榜单补发", "预览动量榜单"):
            button = ttk.Button(parent, text=name, style="Action.TButton", command=lambda task=name: self.run_task(task))
            button.pack(fill="x", pady=5)
            self.action_buttons.append(button)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="运行日志", style="PanelTitle.TLabel").pack(anchor="w")
        self.log_text = tk.Text(
            parent,
            wrap="word",
            bg="#161b22",
            fg="#d1d5db",
            insertbackground="#d1d5db",
            relief="flat",
            font=("Menlo", 12),
        )
        self.log_text.pack(fill="both", expand=True, pady=(12, 0))
        self.log_text.insert("end", "桌面程序已就绪。\n")
        self.log_text.configure(state="disabled")

    def _build_side_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="常用入口", style="PanelTitle.TLabel").pack(anchor="w")
        links = [
            ("打开 core 输出目录", CORE_OUTPUT_DIR),
            ("打开 monitor 输出目录", MONITOR_OUTPUT_DIR),
            ("打开 nav 图", CORE_OUTPUT_DIR / "nav_with_trades.png"),
            ("打开近6个月 nav 图", CORE_OUTPUT_DIR / "recent_6m_nav_with_trades.png"),
            ("打开策略对比图", CORE_OUTPUT_DIR / "recent_6m_strategy_vs_hs300_nasdaq.png"),
            ("打开 trades.csv", CORE_OUTPUT_DIR / "trades.csv"),
        ]
        for label, path in links:
            ttk.Button(parent, text=label, style="Link.TButton", command=lambda p=path: self.open_path(p)).pack(fill="x", pady=4)

        ttk.Label(parent, text="说明", style="PanelTitle.TLabel").pack(anchor="w", pady=(20, 6))
        notes = (
            "正式通知补发会使用 --force。\n"
            "预览通知/榜单不会覆盖去重状态。\n"
            "如果你想双击启动，用 scripts/open_momentum_desktop.command。"
        )
        note_widget = tk.Text(
            parent,
            height=8,
            wrap="word",
            bg="#fbfaf6",
            fg="#374151",
            relief="flat",
            font=("Helvetica Neue", 11),
        )
        note_widget.insert("1.0", notes)
        note_widget.configure(state="disabled")
        note_widget.pack(fill="x", pady=(4, 0))

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log(self, text: str) -> None:
        self.append_log(text.rstrip("\n") + "\n")

    def refresh_summary(self) -> None:
        row = read_last_csv_row(CORE_OUTPUT_DIR / "backtest_nav.csv")
        code_map = read_selected_name_map(CORE_OUTPUT_DIR / "selected_etfs.csv")
        if row is None:
            for variable in self.summary_vars.values():
                variable.set("--")
            self.run_state_var.set("未找到 output/core/backtest_nav.csv")
            return

        signal_code = str(row.get("signal", "")).strip()
        holding_code = str(row.get("holding", "")).strip()
        signal_text = self._format_code_name(signal_code, code_map)
        holding_text = self._format_code_name(holding_code, code_map)

        self.summary_vars["date"].set(row.get("date", "--"))
        self.summary_vars["nav"].set(format_float(row.get("nav"), digits=4))
        self.summary_vars["drawdown"].set(format_pct(row.get("drawdown"), digits=2))
        self.summary_vars["signal"].set(signal_text)
        self.summary_vars["holding"].set(holding_text)
        self.summary_vars["target_exposure"].set(format_pct(row.get("target_exposure"), digits=1))
        momentum_value = row.get("signal_asset_momentum") or row.get("current_momentum")
        self.summary_vars["momentum"].set(format_pct(momentum_value, digits=2))

    def _format_code_name(self, code: str, code_map: dict[str, str]) -> str:
        if not code:
            return "--"
        name = code_map.get(code, "")
        return f"{code} {name}".strip()

    def set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.action_buttons:
            button.configure(state=state)

    def run_task(self, task_name: str) -> None:
        with self.process_lock:
            if self.active_process is not None:
                messagebox.showinfo("任务进行中", f"当前正在执行：{self.active_task_name}")
                return
            argv = TASKS[task_name]
            env = os.environ.copy()
            env["PYTHONPATH"] = PYTHONPATH_VALUE if not env.get("PYTHONPATH") else f"{PYTHONPATH_VALUE}:{env['PYTHONPATH']}"
            command = [PYTHON_BIN, *argv]
            self.log(f"\n[{task_name}] 启动: {' '.join(command)}")
            self.run_state_var.set(f"运行中: {task_name}")
            self.set_buttons_enabled(False)
            worker = threading.Thread(
                target=self._run_process_worker,
                args=(task_name, command, env),
                daemon=True,
            )
            worker.start()

    def _run_process_worker(self, task_name: str, command: list[str], env: dict[str, str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.log_queue.put(("line", f"[{task_name}] 启动失败: {exc}\n"))
            self.log_queue.put(("done", task_name))
            return

        with self.process_lock:
            self.active_process = process
            self.active_task_name = task_name

        assert process.stdout is not None
        for line in process.stdout:
            self.log_queue.put(("line", line))
        return_code = process.wait()
        self.log_queue.put(("line", f"[{task_name}] 退出码: {return_code}\n"))
        self.log_queue.put(("done", task_name))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item_type, payload = self.log_queue.get_nowait()
                if item_type == "line":
                    self.append_log(payload)
                elif item_type == "done":
                    with self.process_lock:
                        self.active_process = None
                        self.active_task_name = None
                    self.set_buttons_enabled(True)
                    self.refresh_summary()
                    self.run_state_var.set(f"空闲，最近完成: {payload}")
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log_queue)

    def open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("路径不存在", f"未找到:\n{path}")
            return
        subprocess.Popen(["open", str(path)])


def main() -> int:
    root = tk.Tk()
    app = MomentumDesktopApp(root)
    app.log("仓库目录: " + str(REPO_ROOT))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

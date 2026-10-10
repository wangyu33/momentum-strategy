#!/usr/bin/env python3
"""macOS sleep/display guard desktop utility."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PLIST = REPO_ROOT / "artifacts" / "com.codex.caffeinate.plist"
LAUNCH_AGENT_LABEL = "com.codex.caffeinate"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def run_command(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def read_idle_time() -> str:
    result = run_command(["defaults", "-currentHost", "read", "com.apple.screensaver", "idleTime"])
    if result.returncode != 0:
        return "未设置"
    value = result.stdout.strip()
    if not value:
        return "未设置"
    try:
        seconds = int(float(value))
    except ValueError:
        return value
    if seconds == 0:
        return "0 秒（已关闭自动屏保）"
    minutes = seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)} 分钟"
    return f"{seconds} 秒"


def read_pmset_value(key: str) -> str:
    result = run_command(["pmset", "-g"])
    if result.returncode != 0:
        return "--"
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith(key):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return " ".join(parts[1:])
    return "--"


def is_caffeinate_running() -> bool:
    result = run_command(["pgrep", "-x", "caffeinate"])
    return result.returncode == 0 and bool(result.stdout.strip())


def is_launch_agent_installed() -> bool:
    return LAUNCH_AGENT_PATH.exists()


def is_launch_agent_loaded() -> bool:
    result = run_command(["launchctl", "print", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"])
    return result.returncode == 0


class SleepGuardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sleep Guard")
        self.root.geometry("980x700")
        self.root.minsize(900, 640)

        self.status_vars = {
            "process": tk.StringVar(value="--"),
            "autostart": tk.StringVar(value="--"),
            "idle_time": tk.StringVar(value="--"),
            "displaysleep": tk.StringVar(value="--"),
            "sleep": tk.StringVar(value="--"),
        }

        self._configure_style()
        self._build_ui()
        self.refresh_status()

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#efe7db")
        style.configure("App.TFrame", background="#efe7db")
        style.configure("Card.TFrame", background="#faf6ef")
        style.configure("Title.TLabel", background="#efe7db", foreground="#1f2937", font=("Helvetica Neue", 24, "bold"))
        style.configure("Hint.TLabel", background="#efe7db", foreground="#6b7280", font=("Helvetica Neue", 12))
        style.configure("CardTitle.TLabel", background="#faf6ef", foreground="#111827", font=("Helvetica Neue", 14, "bold"))
        style.configure("MetricName.TLabel", background="#faf6ef", foreground="#6b7280", font=("Helvetica Neue", 11))
        style.configure("MetricValue.TLabel", background="#faf6ef", foreground="#111827", font=("Helvetica Neue", 15, "bold"))
        style.configure("Action.TButton", font=("Helvetica Neue", 12, "bold"), padding=(10, 10))
        style.configure("Soft.TButton", font=("Helvetica Neue", 11), padding=(8, 8))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Sleep Guard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="控制 macOS 的 caffeinate 保活、自启和自动屏保设置。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(18, 12))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        status_card = ttk.Frame(top, style="Card.TFrame", padding=18)
        status_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_status_card(status_card)

        action_card = ttk.Frame(top, style="Card.TFrame", padding=18)
        action_card.grid(row=0, column=1, sticky="nsew")
        self._build_action_card(action_card)

        bottom = ttk.Frame(outer, style="App.TFrame")
        bottom.pack(fill="both", expand=True)
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)

        log_card = ttk.Frame(bottom, style="Card.TFrame", padding=18)
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_log_card(log_card)

        help_card = ttk.Frame(bottom, style="Card.TFrame", padding=18)
        help_card.grid(row=0, column=1, sticky="nsew")
        self._build_help_card(help_card)

    def _build_status_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="当前状态", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        items = [
            ("保活进程", "process"),
            ("登录自启", "autostart"),
            ("自动屏保", "idle_time"),
            ("displaysleep", "displaysleep"),
            ("sleep", "sleep"),
        ]
        for idx, (label, key) in enumerate(items, start=1):
            block = ttk.Frame(parent, style="Card.TFrame", padding=(0, 10, 0, 6))
            block.grid(row=idx, column=0, sticky="w")
            ttk.Label(block, text=label, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(block, textvariable=self.status_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))

        ttk.Button(parent, text="刷新状态", style="Soft.TButton", command=self.refresh_status).grid(
            row=len(items) + 1, column=0, sticky="w", pady=(16, 0)
        )

    def _build_action_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="快捷操作", style="CardTitle.TLabel").pack(anchor="w")
        actions = [
            ("开启保活", self.start_caffeinate),
            ("关闭保活", self.stop_caffeinate),
            ("安装登录自启", self.install_autostart),
            ("移除登录自启", self.remove_autostart),
            ("关闭自动屏保", self.disable_screensaver),
            ("恢复 5 分钟屏保", self.restore_screensaver),
        ]
        for label, handler in actions:
            ttk.Button(parent, text=label, style="Action.TButton", command=handler).pack(fill="x", pady=5)

    def _build_log_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="执行日志", style="CardTitle.TLabel").pack(anchor="w")
        self.log_text = tk.Text(
            parent,
            wrap="word",
            bg="#14181f",
            fg="#d1d5db",
            insertbackground="#d1d5db",
            relief="flat",
            font=("Menlo", 12),
        )
        self.log_text.pack(fill="both", expand=True, pady=(12, 0))
        self.append_log("Sleep Guard 已启动。")

    def _build_help_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="说明", style="CardTitle.TLabel").pack(anchor="w")
        text = (
            "1. 开启保活会启动 /usr/bin/caffeinate -dimsu。\n\n"
            "2. 安装登录自启会把 plist 放到 ~/Library/LaunchAgents，并由 launchd 在登录时自动拉起。\n\n"
            "3. 关闭自动屏保会把当前用户的 com.apple.screensaver idleTime 设为 0。\n\n"
            "4. 如果你只想临时防黑屏，通常只需要点“开启保活”。"
        )
        note = tk.Text(
            parent,
            wrap="word",
            height=16,
            bg="#faf6ef",
            fg="#374151",
            relief="flat",
            font=("Helvetica Neue", 11),
        )
        note.insert("1.0", text)
        note.configure(state="disabled")
        note.pack(fill="both", expand=True, pady=(10, 0))

    def append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip("\n") + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def refresh_status(self) -> None:
        self.status_vars["process"].set("运行中" if is_caffeinate_running() else "未运行")
        if is_launch_agent_installed():
            auto_text = "已安装"
            if is_launch_agent_loaded():
                auto_text += " / 已加载"
            else:
                auto_text += " / 未加载"
        else:
            auto_text = "未安装"
        self.status_vars["autostart"].set(auto_text)
        self.status_vars["idle_time"].set(read_idle_time())
        self.status_vars["displaysleep"].set(read_pmset_value("displaysleep"))
        self.status_vars["sleep"].set(read_pmset_value("sleep"))
        self.append_log("状态已刷新。")

    def start_caffeinate(self) -> None:
        if is_caffeinate_running():
            self.append_log("caffeinate 已在运行，跳过重复启动。")
            self.refresh_status()
            return
        process = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dimsu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.append_log(f"已启动 caffeinate，pid={process.pid}。")
        self.refresh_status()

    def stop_caffeinate(self) -> None:
        result = run_command(["pkill", "-x", "caffeinate"])
        if result.returncode == 0:
            self.append_log("已停止 caffeinate。")
        else:
            self.append_log("没有发现正在运行的 caffeinate。")
        self.refresh_status()

    def install_autostart(self) -> None:
        if not ARTIFACT_PLIST.exists():
            messagebox.showerror("缺少 plist", f"未找到:\n{ARTIFACT_PLIST}")
            return
        LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ARTIFACT_PLIST, LAUNCH_AGENT_PATH)
        run_command(["launchctl", "bootout", f"gui/{os.getuid()}", str(LAUNCH_AGENT_PATH)])
        bootstrap = run_command(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(LAUNCH_AGENT_PATH)])
        if bootstrap.returncode != 0:
            messagebox.showerror("安装失败", bootstrap.stderr.strip() or bootstrap.stdout.strip() or "launchctl bootstrap failed")
            self.append_log("安装登录自启失败。")
            self.refresh_status()
            return
        run_command(["launchctl", "enable", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"])
        run_command(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"])
        self.append_log(f"已安装登录自启: {LAUNCH_AGENT_PATH}")
        self.refresh_status()

    def remove_autostart(self) -> None:
        if LAUNCH_AGENT_PATH.exists():
            run_command(["launchctl", "bootout", f"gui/{os.getuid()}", str(LAUNCH_AGENT_PATH)])
            LAUNCH_AGENT_PATH.unlink(missing_ok=True)
            self.append_log("已移除登录自启。")
        else:
            self.append_log("登录自启原本就不存在。")
        self.refresh_status()

    def disable_screensaver(self) -> None:
        result = run_command(["defaults", "-currentHost", "write", "com.apple.screensaver", "idleTime", "-int", "0"])
        if result.returncode != 0:
            messagebox.showerror("设置失败", result.stderr.strip() or result.stdout.strip() or "defaults write failed")
            return
        self.append_log("已关闭自动屏保（idleTime=0）。")
        self.refresh_status()

    def restore_screensaver(self) -> None:
        result = run_command(["defaults", "-currentHost", "write", "com.apple.screensaver", "idleTime", "-int", "300"])
        if result.returncode != 0:
            messagebox.showerror("设置失败", result.stderr.strip() or result.stdout.strip() or "defaults write failed")
            return
        self.append_log("已恢复自动屏保为 5 分钟。")
        self.refresh_status()


def main() -> int:
    root = tk.Tk()
    SleepGuardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

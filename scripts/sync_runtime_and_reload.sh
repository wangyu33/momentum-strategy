#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/Users/bytedance/Documents/trae_projects/wy_test"
RUNTIME_DIR="$HOME/Library/Caches/wy_test_runtime"
RUNTIME_VENV="$RUNTIME_DIR/.venv"
RUNTIME_PYTHON="$RUNTIME_VENV/bin/python"
RUNTIME_REQUIREMENTS_REL="momentum_backtest/runtime-requirements.txt"
BOOTSTRAP_PYTHON="/opt/homebrew/bin/python3"
PIP_CACHE_DIR="$RUNTIME_DIR/.pip-cache"
STRICT_RUNTIME_VALIDATION="${STRICT_RUNTIME_VALIDATION:-0}"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
MONITOR_PLIST="$LAUNCH_AGENTS_DIR/com.codex.etf-momentum-monitor.plist"
BOARD_PLIST="$LAUNCH_AGENTS_DIR/com.codex.etf-momentum-board.plist"
BACKFILL_PLIST="$LAUNCH_AGENTS_DIR/com.codex.momentum-backfill.plist"
GUI_UID="$(id -u)"

ensure_runtime_python() {
  local requirements_file="$RUNTIME_DIR/$RUNTIME_REQUIREMENTS_REL"
  local stamp_file="$RUNTIME_VENV/.bootstrap-fingerprint"
  local requirements_sha python_id desired_fingerprint current_fingerprint

  if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
    printf 'bootstrap python not found: %s\n' "$BOOTSTRAP_PYTHON" >&2
    exit 1
  fi

  mkdir -p "$PIP_CACHE_DIR"

  if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    printf '[2/9] Create runtime venv with %s\n' "$BOOTSTRAP_PYTHON"
    "$BOOTSTRAP_PYTHON" -m venv "$RUNTIME_VENV"
  else
    printf '[2/9] Reuse runtime venv: %s\n' "$RUNTIME_VENV"
  fi

  if [[ ! -f "$requirements_file" ]]; then
    printf 'runtime requirements file not found: %s\n' "$requirements_file" >&2
    exit 1
  fi

  requirements_sha="$(shasum -a 256 "$requirements_file" | awk '{print $1}')"
  python_id="$($RUNTIME_PYTHON - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
  desired_fingerprint=$(printf 'python=%s\nrequirements_sha=%s\n' "$python_id" "$requirements_sha")
  current_fingerprint="$(cat "$stamp_file" 2>/dev/null || true)"

  if [[ "$current_fingerprint" != "$desired_fingerprint" ]]; then
    printf '[3/9] Install runtime dependencies\n'
    PIP_CACHE_DIR="$PIP_CACHE_DIR" "$RUNTIME_PYTHON" -m pip install --upgrade pip
    PIP_CACHE_DIR="$PIP_CACHE_DIR" "$RUNTIME_PYTHON" -m pip install --requirement "$requirements_file"
    printf '%s' "$desired_fingerprint" > "$stamp_file"
  else
    printf '[3/9] Runtime dependencies already up to date\n'
  fi
}

printf '[1/9] Sync project to runtime: %s -> %s\n' "$SRC_DIR" "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pycache' \
  --exclude '.pycache_local' \
  "$SRC_DIR/" "$RUNTIME_DIR/"

printf '[1.5/9] Prepare runtime log paths\n'
mkdir -p \
  "$RUNTIME_DIR/momentum_backtest/output" \
  "$RUNTIME_DIR/momentum_backtest/output/core" \
  "$RUNTIME_DIR/momentum_backtest/output/monitor"
touch \
  "$RUNTIME_DIR/momentum_backtest/output/daily_monitor.stdout.log" \
  "$RUNTIME_DIR/momentum_backtest/output/daily_monitor.stderr.log" \
  "$RUNTIME_DIR/momentum_backtest/output/daily_momentum_board.stdout.log" \
  "$RUNTIME_DIR/momentum_backtest/output/daily_momentum_board.stderr.log" \
  "$RUNTIME_DIR/momentum_backtest/output/run_backtest.stdout.log" \
  "$RUNTIME_DIR/momentum_backtest/output/run_backtest.stderr.log" \
  "$RUNTIME_DIR/momentum_backtest/output/monitor/daily_monitor.run.log" \
  "$RUNTIME_DIR/momentum_backtest/output/monitor/daily_momentum_board.run.log"

ensure_runtime_python

printf '[4/9] Validate runtime outputs\n'
if ! "$RUNTIME_PYTHON" "$RUNTIME_DIR/momentum_backtest/validate_strategy_outputs.py"; then
  if [[ "$STRICT_RUNTIME_VALIDATION" == "1" ]]; then
    exit 1
  fi
  printf '[warn] validate_strategy_outputs.py reported failures; continuing scheduler reload because runtime bootstrap succeeded. Set STRICT_RUNTIME_VALIDATION=1 to fail fast.\n' >&2
fi

printf '[5/9] Rewrite LaunchAgent plist files\n'
cat > "$MONITOR_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.codex.etf-momentum-monitor</string>
    <key>ProgramArguments</key>
    <array>
      <string>$RUNTIME_PYTHON</string>
      <string>$RUNTIME_DIR/momentum_backtest/daily_monitor.py</string>
      <string>--disable-direct-feishu</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$RUNTIME_DIR</string>
    <key>StartCalendarInterval</key>
    <array>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>14</integer><key>Minute</key><integer>50</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>14</integer><key>Minute</key><integer>50</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>14</integer><key>Minute</key><integer>50</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>14</integer><key>Minute</key><integer>50</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>14</integer><key>Minute</key><integer>50</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/daily_monitor.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/daily_monitor.stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
  </dict>
</plist>
PLIST

cat > "$BOARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.codex.etf-momentum-board</string>
    <key>ProgramArguments</key>
    <array>
      <string>$RUNTIME_PYTHON</string>
      <string>$RUNTIME_DIR/momentum_backtest/daily_momentum_board.py</string>
      <string>--disable-direct-feishu</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$RUNTIME_DIR</string>
    <key>StartCalendarInterval</key>
    <array>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>10</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>10</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/daily_momentum_board.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/daily_momentum_board.stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
  </dict>
</plist>
PLIST

cat > "$BACKFILL_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.codex.momentum-backfill</string>
    <key>ProgramArguments</key>
    <array>
      <string>$RUNTIME_PYTHON</string>
      <string>$RUNTIME_DIR/momentum_backtest/run_backtest.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$RUNTIME_DIR</string>
    <key>StartCalendarInterval</key>
    <array>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/run_backtest.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$RUNTIME_DIR/momentum_backtest/output/run_backtest.stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
  </dict>
</plist>
PLIST

printf '[6/9] Reload LaunchAgents\n'
launchctl bootout "gui/$GUI_UID/com.codex.etf-momentum-monitor" >/dev/null 2>&1 || true
launchctl bootout "gui/$GUI_UID/com.codex.etf-momentum-board" >/dev/null 2>&1 || true
launchctl bootout "gui/$GUI_UID/com.codex.momentum-backfill" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$GUI_UID" "$MONITOR_PLIST"
launchctl bootstrap "gui/$GUI_UID" "$BOARD_PLIST"
launchctl bootstrap "gui/$GUI_UID" "$BACKFILL_PLIST"

printf '[7/9] Catch up missed slots after reload if needed\n'
STAMP_DIR="$RUNTIME_DIR/momentum_backtest/output/monitor/scheduler_catchup"
mkdir -p "$STAMP_DIR"
TODAY_YYYYMMDD="$(date +%Y%m%d)"
TODAY_WEEKDAY="$(date +%u)"
NOW_HHMM="$(date +%H%M)"
MONITOR_STATE="$RUNTIME_DIR/momentum_backtest/output/monitor/daily_monitor_state.json"
BOARD_STATE="$RUNTIME_DIR/momentum_backtest/output/monitor/daily_momentum_board_state.json"
BACKTEST_NAV="$RUNTIME_DIR/momentum_backtest/output/core/backtest_nav.csv"

file_mtime_same_day_after() {
  local target="$1"
  local slot_hhmm="$2"
  if [[ ! -f "$target" ]]; then
    return 1
  fi
  local file_day file_hhmm
  file_day="$(date -r "$target" +%Y%m%d 2>/dev/null || true)"
  file_hhmm="$(date -r "$target" +%H%M 2>/dev/null || true)"
  [[ "$file_day" == "$TODAY_YYYYMMDD" && "$file_hhmm" > "$slot_hhmm" || "$file_hhmm" == "$slot_hhmm" ]]
}

maybe_catchup_slot() {
  local service="$1"
  local slot_name="$2"
  local slot_hhmm="$3"
  local evidence_file="$4"
  local stamp_file="$STAMP_DIR/${service}_${slot_name}_${TODAY_YYYYMMDD}.stamp"

  if [[ "$TODAY_WEEKDAY" -gt 5 ]]; then
    return 0
  fi
  if [[ "$NOW_HHMM" < "$slot_hhmm" ]]; then
    return 0
  fi
  if [[ -f "$stamp_file" ]]; then
    return 0
  fi
  if file_mtime_same_day_after "$evidence_file" "$slot_hhmm"; then
    return 0
  fi

  printf '  - catch up %s %s\n' "$service" "$slot_name"
  launchctl kickstart -k "gui/$GUI_UID/${service}"
  : > "$stamp_file"
}

maybe_catchup_slot "com.codex.etf-momentum-monitor" "0940" "0940" "$MONITOR_STATE"
maybe_catchup_slot "com.codex.etf-momentum-monitor" "1210" "1210" "$MONITOR_STATE"
maybe_catchup_slot "com.codex.etf-momentum-monitor" "1450" "1450" "$MONITOR_STATE"
maybe_catchup_slot "com.codex.etf-momentum-board" "1510" "1510" "$BOARD_STATE"
maybe_catchup_slot "com.codex.momentum-backfill" "1520" "1520" "$BACKTEST_NAV"

printf '[8/9] Print registered jobs\n'
launchctl print "gui/$GUI_UID/com.codex.etf-momentum-monitor" | sed -n '1,25p'
printf '\n'
launchctl print "gui/$GUI_UID/com.codex.etf-momentum-board" | sed -n '1,25p'
printf '\n'
launchctl print "gui/$GUI_UID/com.codex.momentum-backfill" | sed -n '1,25p'

printf '[9/9] Done. Runtime sync + scheduler reload complete.\n'

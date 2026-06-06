#!/usr/bin/env -S .venv/bin/python
"""
Hand Cursor — TUI Dashboard

Интерактивный интерфейс с живыми данными с камеры и настройками.

Запуск:
    python3 cli.py              — full dashboard + pipeline
    python3 cli.py --no-rust    — dashboard без Rust-драйвера
    python3 cli.py show         — показать настройки (JSON)
    python3 cli.py set <k> <v>  — изменить настройку
    python3 cli.py defaults     — сброс
"""

import json
import os
import select
import signal
import time
import subprocess
import sys
import termios
import threading
import tty
from pathlib import Path

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ─── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
SETTINGS_PATH = ROOT / "settings.json"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
RUST_BIN = ROOT / "target" / "debug" / "open-tracker"

# ─── defaults ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "mode": "tilt",
    "sensitivity_x": 2.0,
    "sensitivity_y": 1.5,
    "smooth": 0.3,
    "dead_zone": 0.03,
    "confidence": 0.7,
    "edge_boost": 10.0,
    "pinch_threshold": 0.05,
}

MODES = ["tilt", "tilt-vector", "delta", "position"]
MODE_DESC = {
    "tilt": "наклон костяшек + длина",
    "tilt-vector": "вектор запястье→ладонь",
    "delta": "магнит: движение → курсор",
    "position": "смещение от центра",
}
MODE_GLYPH = {m: "○" for m in MODES}

# настройки, которые можно менять стрелками
ADJUSTABLE_KEYS = [
    "sensitivity_x",
    "sensitivity_y",
    "smooth",
    "dead_zone",
    "confidence",
    "edge_boost",
]
ADJUSTABLE_RANGES = {
    "sensitivity_x": (0.1, 10.0),
    "sensitivity_y": (0.1, 10.0),
    "smooth": (0.01, 1.0),
    "dead_zone": (0.0, 0.3),
    "confidence": (0.1, 1.0),
    "edge_boost": (0.0, 50.0),
}

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
#  Settings
# ═══════════════════════════════════════════════════════════════════════════════

def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return {**DEFAULTS, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULTS)


def save_settings(s: dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)
        f.write("\n")


def clean_settings(s: dict) -> dict:
    """Гарантирует что все ключи есть."""
    return {**DEFAULTS, **{k: v for k, v in s.items() if k in DEFAULTS}}

# ═══════════════════════════════════════════════════════════════════════════════
#  Keyboard raw-mode
# ═══════════════════════════════════════════════════════════════════════════════

class TerminalInput:
    """Reads keys one at a time without requiring Enter."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = None
        self._buf = ""
        self._is_tty = sys.stdin.isatty()

    def __enter__(self):
        if self._is_tty:
            try:
                self._old = termios.tcgetattr(self._fd)
                mode = termios.tcgetattr(self._fd)
                # ── input: raw (no line buffer, no echo, no signals) ──
                mode[tty.IFLAG] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
                mode[tty.LFLAG] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
                mode[tty.CC][termios.VMIN] = 1
                mode[tty.CC][termios.VTIME] = 0
                # ── output: keep OPOST (Rich needs \n → \r\n) ──
                # ── cflags: keep default ──
                termios.tcsetattr(self._fd, termios.TCSADRAIN, mode)
            except termios.error:
                self._is_tty = False
        return self

    def __exit__(self, *args):
        if self._is_tty and self._old:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except termios.error:
                pass

    # ── helpers ────────────────────────────────────────────────────────────────

    def _read_buf(self) -> str | None:
        """Extract one keypress from buffer, handling escape sequences."""
        if not self._buf:
            return None

        ch = self._buf[0]

        # Arrow keys: [A [B [C [D (3-byte sequences)
        if ch == "" and len(self._buf) >= 3:
            seq = self._buf[:3]
            if seq[1] == "[" and seq[2] in ("A", "B", "C", "D"):
                self._buf = self._buf[3:]
                return seq

        # Single character (including lone ESC)
        self._buf = self._buf[1:]
        return ch

    # ── poll ───────────────────────────────────────────────────────────────────

    def poll(self) -> str | None:
        """Return next keypress or None if nothing available."""
        if not self._is_tty:
            return None

        # First drain any buffered data from previous reads
        if self._buf:
            return self._read_buf()

        # Read fresh bytes from terminal (non-blocking)
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                raw = os.read(self._fd, 8).decode("utf-8", errors="replace")
                if raw:
                    self._buf = raw
                    return self._read_buf()
        except (OSError, ValueError):
            pass
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Запускает vision и опционально Rust-драйвер."""

    def __init__(self, use_rust: bool = True):
        self._vision: subprocess.Popen | None = None
        self._rust: subprocess.Popen | None = None
        self._use_rust = use_rust and RUST_BIN.exists()
        self._latest: dict | None = None
        self._running = True

    @property
    def latest(self) -> dict | None:
        return self._latest

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if not VENV_PYTHON.exists():
            print("✗ .venv/bin/python не найден. Запусти: .venv/bin/pip install -r requirements.txt")
            self._running = False
            return

        self._vision = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "vision.main"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if self._use_rust:
            self._rust = subprocess.Popen(
                [str(RUST_BIN)],
                stdin=self._vision.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # читаем JSON в потоке
        def _reader():
            buf = ""
            while self._running and self._vision and self._vision.stdout:
                try:
                    chunk = self._vision.stdout.read1(4096).decode()
                    if not chunk:
                        break
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                self._latest = json.loads(line)
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if self._rust:
            try:
                self._rust.terminate()
                self._rust.wait(timeout=3)
            except Exception:
                self._rust.kill()
        if self._vision:
            try:
                self._vision.terminate()
                self._vision.wait(timeout=3)
            except Exception:
                self._vision.kill()

# ═══════════════════════════════════════════════════════════════════════════════
#  Render helpers
# ═══════════════════════════════════════════════════════════════════════════════

BAR_CHARS = "▏▎▍▌▋▊▉█"

def text_bar(value: float, width: int = 16) -> Text:
    """Рисует прогресс-бар из символов."""
    n = int(value * width * 8)
    full = n // 8
    part = n % 8
    bar = "█" * full
    if part:
        bar += BAR_CHARS[part - 1]
    bar += "░" * (width - full - (1 if part else 0))
    return Text(bar)


def make_dashboard(settings: dict, data: dict | None, selected: int) -> Layout:
    mode = settings.get("mode", "tilt")
    s = settings

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # ── header ────────────────────────────────────────────────────────────────
    title = Text("🖐  Hand Cursor", style="bold cyan")
    title += Text("  —  управление камерой", style="dim white")
    layout["header"].update(Panel(title, style="bold"))

    # ── body ──────────────────────────────────────────────────────────────────
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )

    # -- left: Live data --
    live_lines = []

    # режим
    mode_text = Text("Mode:  ", style="bold")
    for m in MODES:
        if m == mode:
            mode_text += Text(f" ● {m}  ", style="green bold")
        else:
            mode_text += Text(f" ○ {m}  ", style="dim")
    live_lines.append(mode_text)
    live_lines.append(Text())

    # X bar
    if data:
        x = data.get("x", 0)
        y = data.get("y", 0)
        btn = data.get("button", False)

        xn = max(0, min(1, (x + 127) / 254))
        yn = max(0, min(1, (y + 127) / 254))

        x_bar = Text(" X  ")
        x_bar += text_bar(xn)
        x_bar += Text(f"  {x:+4d}", style="cyan bold")
        live_lines.append(x_bar)

        y_bar = Text(" Y  ")
        y_bar += text_bar(yn)
        y_bar += Text(f"  {y:+4d}", style="magenta bold")
        live_lines.append(y_bar)

        btn_style = "green" if not btn else "red"
        btn_txt = "● Pressed" if btn else "● Released"
        live_lines.append(Text(f" 🖱  {btn_txt}", style=btn_style))
    else:
        live_lines.append(Text(" ⏳ Waiting for camera...", style="dim italic"))

    live_lines.append(Text())
    live_lines.append(Text(f" mode: {mode}  —  {MODE_DESC.get(mode, '')}", style="italic"))

    layout["body"]["left"].update(
        Panel(Group(*live_lines), title="Live", border_style="cyan")
    )

    # -- right: Settings --
    setting_lines = []
    for i, key in enumerate(ADJUSTABLE_KEYS):
        val = s.get(key, DEFAULTS.get(key, 0))
        rmin, rmax = ADJUSTABLE_RANGES.get(key, (0, 1))
        vn = max(0, min(1, (val - rmin) / (rmax - rmin))) if rmax > rmin else 0

        prefix = "▸ " if i == selected else "  "
        label = f"{prefix}{key:<16}"
        label_style = "bold green" if i == selected else ""

        bar_w = text_bar(vn, width=12)
        val_str = f"{val:<6.2f}"
        if i == selected:
            val_str = f"◄ {val:<5.2f} ►"

        line = Text(label, style=label_style)
        line += bar_w
        line += Text(f" {val_str}", style="bold" if i == selected else "")
        setting_lines.append(line)

    layout["body"]["right"].update(
        Panel(Group(*setting_lines), title="Settings", border_style="yellow")
    )

    # ── footer ────────────────────────────────────────────────────────────────
    footer = Text()
    footer += Text(" [Q]uit", style="bold")
    footer += Text("  [M]ode", style="bold")
    footer += Text("  [R]eset", style="bold")
    footer += Text("  [↑][↓] select  [←][→] adjust", style="dim")
    layout["footer"].update(
        Panel(footer, style="dim")
    )

    return layout


# ═══════════════════════════════════════════════════════════════════════════════
#  Dashboard loop
# ═══════════════════════════════════════════════════════════════════════════════

KEY_MODES = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "q": "quit",
    "Q": "quit",
    "m": "mode",
    "M": "mode",
    "r": "reset",
    "R": "reset",
}


def run_dashboard(use_rust: bool = True):
    if not sys.stdin.isatty():
        print("Dashboard requires an interactive terminal (TTY).")
        print("Use: python3 cli.py show / set / defaults")
        return
    pipe = Pipeline(use_rust=use_rust)
    pipe.start()

    settings = clean_settings(load_settings())
    selected = 0
    running = True

    def _make_layout():
        return make_dashboard(settings, pipe.latest, selected)

    try:
        with TerminalInput() as tinput:
            with Live(_make_layout(), console=console, refresh_per_second=15, screen=False) as live:
                while running:
                    # ── check keyboard ──
                    while True:
                        k = tinput.poll()
                        if k is None:
                            break
                        cmd = KEY_MODES.get(k)
                        if cmd == "quit":
                            running = False
                        elif cmd == "up":
                            selected = max(0, selected - 1)
                        elif cmd == "down":
                            selected = min(len(ADJUSTABLE_KEYS) - 1, selected + 1)
                        elif cmd == "mode":
                            modes = MODES
                            cur = modes.index(settings["mode"]) if settings.get("mode") in modes else 0
                            settings["mode"] = modes[(cur + 1) % len(modes)]
                            save_settings(settings)
                        elif cmd == "reset":
                            settings = dict(DEFAULTS)
                            save_settings(settings)
                        elif cmd == "right":
                            key = ADJUSTABLE_KEYS[selected]
                            rmin, rmax = ADJUSTABLE_RANGES.get(key, (0, 1))
                            delta = (rmax - rmin) * 0.02
                            settings[key] = min(rmax, settings.get(key, 0) + delta)
                            save_settings(settings)
                        elif cmd == "left":
                            key = ADJUSTABLE_KEYS[selected]
                            rmin, rmax = ADJUSTABLE_RANGES.get(key, (0, 1))
                            delta = (rmax - rmin) * 0.02
                            settings[key] = max(rmin, settings.get(key, 0) - delta)
                            save_settings(settings)

                    # ── reload settings (на случай внешних изменений) ──
                    file_settings = load_settings()
                    if file_settings.get("mode") != settings.get("mode"):
                        settings["mode"] = file_settings.get("mode", "tilt")

                    # ── update ──
                    live.update(_make_layout())
                    time.sleep(0.016)

    finally:
        pipe.stop()
        print()  # clean exit


# ═══════════════════════════════════════════════════════════════════════════════
#  Direct commands (старый режим)
# ═══════════════════════════════════════════════════════════════════════════════

def direct(args):
    if not args:
        run_dashboard()
        return

    cmd = args[0]

    if cmd == "show":
        data = load_settings()
        print(json.dumps(data, indent=2))
        return

    if cmd == "defaults":
        save_settings(dict(DEFAULTS))
        print("✓ Settings reset to defaults")
        return

    if cmd in ("-h", "--help", "help"):
        print(__doc__.strip())
        return

    if cmd == "set" and len(args) >= 3:
        key = args[1]
        val = args[2]
        s = load_settings()

        if key == "mode":
            if val in MODES:
                s["mode"] = val
                print(f"✓ mode = {val}")
            else:
                print(f"✗ Unknown mode: {val}")
                return
        else:
            try:
                s[key] = float(val)
                print(f"✓ {key} = {val}")
            except ValueError:
                print(f"✗ Not a number: {val}")
                return
        save_settings(s)
        return

    if cmd == "--no-rust":
        if not sys.stdin.isatty():
            print("Dashboard requires a TTY. Use: show / set / defaults")
            return
        run_dashboard(use_rust=False)
        return

    print("Usage: cli.py [show | defaults | set <key> <val> | --no-rust]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    direct(sys.argv[1:])

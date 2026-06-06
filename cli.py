#!/usr/bin/env python3
"""Interactive CLI for hand-cursor settings.

Usage:
    python cli.py              # interactive mode
    python cli.py set mode delta   # quick set
    python cli.py set sensitivity_x 3.0
    python cli.py show             # show current settings
"""

import json
import os
import shutil
import sys
from pathlib import Path

SETTINGS_PATH = Path(__file__).parent / "settings.json"

# ── defaults ──────────────────────────────────────────────────────────────────

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

# ── helpers ───────────────────────────────────────────────────────────────────

def load() -> dict:
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            return {**DEFAULTS, **json.load(f)}
    return dict(DEFAULTS)


def save(settings: dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")


# ── output ────────────────────────────────────────────────────────────────────

def show_table():
    s = load()
    tw = shutil.get_terminal_size().columns
    width = min(tw - 4, 50)

    print()
    print("╭" + "─" * width + "╮")
    print(f"│{' Hand Cursor — Settings':<{width}}│")
    print("├" + "─" * width + "┤")

    print(f"│ {'mode':<20} {s.get('mode', '?'):<{width-23}}│")
    print(f"│ {'sensitivity_x':<20} {s.get('sensitivity_x', '?'):<{width-23}}│")
    print(f"│ {'sensitivity_y':<20} {s.get('sensitivity_y', '?'):<{width-23}}│")
    print(f"│ {'smooth':<20} {s.get('smooth', '?'):<{width-23}}│")
    print(f"│ {'dead_zone':<20} {s.get('dead_zone', '?'):<{width-23}}│")
    print(f"│ {'confidence':<20} {s.get('confidence', '?'):<{width-23}}│")
    print(f"│ {'edge_boost':<20} {s.get('edge_boost', '?'):<{width-23}}│")
    print(f"│ {'pinch_threshold':<20} {s.get('pinch_threshold', '?'):<{width-23}}│")

    print("├" + "─" * width + "┤")
    print(f"│ {'Modes:':<{width}}│")
    for m in MODES:
        mark = "●" if s.get("mode") == m else "○"
        desc = {
            "tilt": "наклон костяшек + длина ладони",
            "tilt-vector": "вектор запястье→ладонь",
            "delta": "магнит: движение руки = курсор",
            "position": "позиция: смещение от центра",
        }
        print(f"│  {mark} {m:<15} {desc.get(m, ''):<{width-20}}│")
    print("╰" + "─" * width + "╯")
    print()


# ── interactive loop ──────────────────────────────────────────────────────────

def interactive():
    show_table()
    print("  Команды:")
    print("    set <key> <value>   — изменить настройку")
    print("    set mode <name>     — переключить режим")
    print("    show                — показать таблицу")
    print("    defaults            — сбросить всё")
    print("    quit / Ctrl+C       — выход")
    print()

    while True:
        try:
            cmd = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        action = parts[0] if parts else ""

        if action == "quit" or action == "exit" or action == "q":
            break

        elif action == "show":
            show_table()

        elif action == "defaults":
            save(dict(DEFAULTS))
            print("✓ Сброшено на заводские настройки\n")
            show_table()

        elif action == "set" and len(parts) >= 3:
            key = parts[1]
            val = parts[2]

            s = load()

            if key == "mode":
                if val in MODES:
                    s["mode"] = val
                    print(f"✓ Режим: {val}")
                else:
                    print(f"✗ Нет такого режима: {val}. Доступны: {', '.join(MODES)}")
                    continue
            else:
                try:
                    s[key] = float(val)
                    print(f"✓ {key} = {val}")
                except ValueError:
                    print(f"✗ Не число: {val}")
                    continue

            save(s)

        elif action == "set" and len(parts) == 2:
            print("✗ Укажи значение: set <key> <value>")

        else:
            print(f"✗ Неизвестная команда: {cmd}")

    print("👋")


# ── direct mode ───────────────────────────────────────────────────────────────

def direct(args):
    if len(args) >= 1 and args[0] == "show":
        data = load()
        print(json.dumps(data, indent=2))
        return

    if len(args) >= 1 and args[0] == "defaults":
        save(dict(DEFAULTS))
        print("✓ Settings reset to defaults")
        return

    if len(args) >= 2 and args[0] == "set":
        key = args[1]
        val = args[2] if len(args) >= 3 else None

        if not val:
            print("Usage: cli.py set <key> <value>")
            return

        s = load()

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

        save(s)
        return

    print("Usage: cli.py [show | defaults | set <key> <value>]")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        direct(sys.argv[1:])
    else:
        interactive()

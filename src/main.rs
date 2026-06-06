use evdev::uinput::VirtualDeviceBuilder;
use evdev::{
    InputEvent, EventType, Synchronization,
    RelativeAxisType, Key,
};
use serde::Deserialize;
use std::io::{self, BufRead};

// ─── Конфигурация ────────────────────────────────────────────────────────────

const DEAD_ZONE: f32 = 0.5;      // минимальная дельта (рука стоит → не двигаем)
const SENSITIVITY: f32 = 1.0;    // множитель скорости (1.0 = как есть из Python)

// ─── Формат команды из Python ────────────────────────────────────────────────

#[derive(Deserialize)]
struct Command {
    x: i32,
    y: i32,
    button: bool,
    #[serde(default)]
    scroll: i32,
    #[serde(default)]
    hscroll: i32,
}

// ─── Виртуальная мышь ────────────────────────────────────────────────────────

fn build_virtual_mouse() -> Result<evdev::uinput::VirtualDevice, Box<dyn std::error::Error>> {
    let device = VirtualDeviceBuilder::new()?
        .name("open-tracker-virtual-mouse")
        .with_relative_axes(&{
            let mut axes = evdev::AttributeSet::new();
            axes.insert(RelativeAxisType::REL_X);
            axes.insert(RelativeAxisType::REL_Y);
            axes.insert(RelativeAxisType::REL_WHEEL);
            axes.insert(RelativeAxisType::REL_HWHEEL);
            axes
        })?
        .with_keys(&{
            let mut keys = evdev::AttributeSet::new();
            keys.insert(Key::BTN_LEFT);
            keys
        })?
        .build()?;

    Ok(device)
}

// ─── Главный цикл ────────────────────────────────────────────────────────────

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = build_virtual_mouse()?;
    let mut btn_pressed = false;

    eprintln!("🖱️  виртуальная мышь создана, жду команды...");

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }

        let cmd: Command = match serde_json::from_str(&text) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("⚠️  не удалось разобрать JSON: {e} — строка: {text}");
                continue;
            }
        };

        // ── движение курсора ──
        let dx = if (cmd.x as f32).abs() < DEAD_ZONE {
            0
        } else {
            (cmd.x as f32 * SENSITIVITY) as i32
        };
        let dy = if (cmd.y as f32).abs() < DEAD_ZONE {
            0
        } else {
            (cmd.y as f32 * SENSITIVITY) as i32
        };

        if dx != 0 || dy != 0 {
            let ev_x = InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_X.0, dx);
            let ev_y = InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_Y.0, dy);
            let ev_syn = InputEvent::new(EventType::SYNCHRONIZATION, Synchronization::SYN_REPORT.0, 0);
            device.emit(&[ev_x, ev_y, ev_syn])?;
        }

        // ── клик ──
        let new_btn = cmd.button;
        if new_btn != btn_pressed {
            let val = if new_btn { 1 } else { 0 };
            let ev_btn = InputEvent::new(EventType::KEY, Key::BTN_LEFT.0, val);
            let ev_syn = InputEvent::new(EventType::SYNCHRONIZATION, Synchronization::SYN_REPORT.0, 0);
            device.emit(&[ev_btn, ev_syn])?;
            btn_pressed = new_btn;
        }

        // ── скролл ──
        if cmd.scroll != 0 || cmd.hscroll != 0 {
            let ev_scroll = InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_WHEEL.0, cmd.scroll);
            let ev_hscroll = InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_HWHEEL.0, cmd.hscroll);
            let ev_syn = InputEvent::new(EventType::SYNCHRONIZATION, Synchronization::SYN_REPORT.0, 0);
            device.emit(&[ev_scroll, ev_hscroll, ev_syn])?;
        }
    }

    Ok(())
}

// ─── Точка входа ─────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("💥 ошибка: {e}");
        std::process::exit(1);
    }
}

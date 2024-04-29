use glam::UVec2;
use std::{
    collections::HashMap,
    hash::DefaultHasher,
    time::{Duration, Instant},
};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseScrollDelta, StartCause, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, KeyLocation, NamedKey},
    platform::pump_events::EventLoopExtPumpEvents,
    raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle},
    window::WindowBuilder,
};

pub use winit::keyboard::SmolStr;

pub struct Window {
    event_loop: EventLoop<()>,
    window: winit::window::Window,
    active_inputs: HashMap<InputKey, f32>,
}

#[derive(Clone, Debug)]
pub enum Event {
    Input(Input),
    Resized(UVec2),
    CloseRequested,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub key: InputKey,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputKey {
    /// Unicode character, case sensitive.
    ///
    /// Includes "enter" (`\n`), "tab" (`\t`) and "space" (` `)
    Unicode(SmolStr),
    LShift,
    RShift,
    LCtrl,
    RCtrl,
    LSuper,
    RSuper,
    Alt,
    AltGr,
    MouseRelativeX,
    MouseRelativeY,
    MouseButtonL,
    MouseButtonM,
    MouseButtonR,
    MouseWheelY,
    PrintScreen,
    PageUp,
    PageDown,
    Esc,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Backspace,
    F(u8),
}

impl Window {
    pub fn new() -> Self {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().build(&event_loop).unwrap();
        Self {
            event_loop,
            window,
            active_inputs: Default::default(),
        }
    }

    pub fn wait(&mut self, until: Instant) -> Vec<Event> {
        let dur = until
            .checked_duration_since(Instant::now())
            .unwrap_or(Duration::ZERO);
        let mut events = Vec::new();
        type E<'a, T> = winit::event::Event<T>;

        for key in [InputKey::MouseRelativeX, InputKey::MouseRelativeY] {
            events.push(Event::Input(Input {
                value: self.input(&key),
                key,
            }));
        }

        self.event_loop
            .pump_events(Some(dur), |event, _| match event {
                E::WindowEvent {
                    window_id: _,
                    event,
                } => match event {
                    WindowEvent::Resized(s) => {
                        events.push(Event::Resized(UVec2::new(s.width, s.height)))
                    }
                    WindowEvent::Ime(_) => {}
                    WindowEvent::Moved(_) => {}
                    WindowEvent::Touch(_) => todo!(),
                    WindowEvent::Destroyed => todo!(),
                    WindowEvent::Focused(_) => {}
                    WindowEvent::Occluded(_) => {}
                    WindowEvent::CursorLeft { device_id } => {}
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                    } => {
                        // FIXME accumulate like rel mouse
                        match delta {
                            MouseScrollDelta::LineDelta(_, y) => {
                                events.push(set_input(
                                    &mut self.active_inputs,
                                    InputKey::MouseWheelY,
                                    y,
                                ));
                            }
                            MouseScrollDelta::PixelDelta(_) => todo!(),
                        }
                    }
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                    } => {}
                    WindowEvent::AxisMotion {
                        device_id,
                        axis,
                        value,
                    } => {}
                    WindowEvent::DroppedFile(_) => todo!(),
                    WindowEvent::HoveredFile(_) => todo!(),
                    WindowEvent::CursorMoved {
                        device_id,
                        position,
                    } => {}
                    WindowEvent::CloseRequested => events.push(Event::CloseRequested),
                    WindowEvent::SmartMagnify { device_id } => todo!(),
                    WindowEvent::ThemeChanged(_) => todo!(),
                    WindowEvent::KeyboardInput {
                        device_id,
                        event,
                        is_synthetic: _,
                    } => {
                        use InputKey::*;
                        let key = match event.logical_key {
                            Key::Named(n) => match (n, event.location) {
                                (NamedKey::Space, _) => Unicode(SmolStr::new_inline(" ")),
                                (NamedKey::Enter, _) => Unicode(SmolStr::new_inline("\n")),
                                (NamedKey::Tab, _) => Unicode(SmolStr::new_inline("\t")),
                                (NamedKey::Control, KeyLocation::Left) => LCtrl,
                                (NamedKey::Control, KeyLocation::Right) => RCtrl,
                                (NamedKey::Shift, KeyLocation::Left) => LShift,
                                (NamedKey::Shift, KeyLocation::Right) => RShift,
                                (NamedKey::Super, KeyLocation::Left) => LSuper,
                                (NamedKey::Super, KeyLocation::Right) => RSuper,
                                (NamedKey::Alt, _) => Alt,
                                (NamedKey::AltGraph, _) => AltGr,
                                (NamedKey::PrintScreen, _) => PrintScreen,
                                (NamedKey::PageUp, _) => PageUp,
                                (NamedKey::PageDown, _) => PageDown,
                                (NamedKey::F1, _) => F(1),
                                (NamedKey::F2, _) => F(2),
                                (NamedKey::F3, _) => F(3),
                                (NamedKey::F4, _) => F(4),
                                (NamedKey::F5, _) => F(5),
                                (NamedKey::F6, _) => F(6),
                                (NamedKey::F7, _) => F(7),
                                (NamedKey::F8, _) => F(8),
                                (NamedKey::F9, _) => F(9),
                                (NamedKey::F10, _) => F(10),
                                (NamedKey::F11, _) => F(11),
                                (NamedKey::F12, _) => F(12),
                                (NamedKey::F13, _) => F(13),
                                (NamedKey::F14, _) => F(14),
                                (NamedKey::F15, _) => F(15),
                                (NamedKey::F16, _) => F(16),
                                (NamedKey::F17, _) => F(17),
                                (NamedKey::F18, _) => F(18),
                                (NamedKey::F19, _) => F(19),
                                (NamedKey::F20, _) => F(20),
                                (NamedKey::F21, _) => F(21),
                                (NamedKey::F22, _) => F(22),
                                (NamedKey::F23, _) => F(23),
                                (NamedKey::F24, _) => F(24),
                                (NamedKey::F25, _) => F(25),
                                (NamedKey::F26, _) => F(26),
                                (NamedKey::F27, _) => F(27),
                                (NamedKey::F28, _) => F(28),
                                (NamedKey::F29, _) => F(29),
                                (NamedKey::F30, _) => F(30),
                                (NamedKey::F31, _) => F(31),
                                (NamedKey::F32, _) => F(32),
                                (NamedKey::F33, _) => F(33),
                                (NamedKey::Escape, _) => Esc,
                                (NamedKey::ArrowUp, _) => ArrowUp,
                                (NamedKey::ArrowDown, _) => ArrowDown,
                                (NamedKey::ArrowLeft, _) => ArrowLeft,
                                (NamedKey::ArrowRight, _) => ArrowRight,
                                (NamedKey::Backspace, _) => Backspace,
                                n => todo!("{:?}", n),
                            },
                            Key::Dead(_) => todo!(),
                            Key::Character(s) => Unicode(s),
                            Key::Unidentified(_) => todo!(),
                        };
                        let value = match event.state {
                            ElementState::Pressed => 1.0,
                            ElementState::Released => 0.0,
                        };
                        events.push(set_input(&mut self.active_inputs, key, value));
                    }
                    WindowEvent::CursorEntered { device_id } => {}
                    WindowEvent::TouchpadRotate {
                        device_id,
                        delta,
                        phase,
                    } => todo!(),
                    WindowEvent::TouchpadMagnify {
                        device_id,
                        delta,
                        phase,
                    } => todo!(),
                    WindowEvent::ModifiersChanged(_) => {}
                    WindowEvent::TouchpadPressure {
                        device_id,
                        pressure,
                        stage,
                    } => todo!(),
                    WindowEvent::HoveredFileCancelled => todo!(),
                    WindowEvent::ScaleFactorChanged {
                        scale_factor,
                        inner_size_writer,
                    } => todo!(),
                    WindowEvent::ActivationTokenDone { serial, token } => todo!(),
                    WindowEvent::RedrawRequested => {}
                },
                E::NewEvents(event) => match event {
                    StartCause::ResumeTimeReached {
                        start,
                        requested_resume,
                    } => todo!(),
                    StartCause::Poll => todo!(),
                    StartCause::Init => {}
                    StartCause::WaitCancelled {
                        start,
                        requested_resume,
                    } => {}
                },
                E::Resumed => {}
                E::Suspended => todo!(),
                E::DeviceEvent { device_id, event } => match event {
                    DeviceEvent::Key(_) => {}
                    DeviceEvent::Motion { axis, value } => {}
                    DeviceEvent::MouseMotion { delta: (x, y) } => {
                        for e in [(InputKey::MouseRelativeX, x), (InputKey::MouseRelativeY, y)] {
                            let v = *self.active_inputs.get(&e.0).unwrap_or(&0.0);
                            events.push(set_input(&mut self.active_inputs, e.0, v + e.1 as f32));
                        }
                    }
                    DeviceEvent::Button { button, state } => {
                        let key = match button {
                            1 => InputKey::MouseButtonL,
                            2 => InputKey::MouseButtonM,
                            3 => InputKey::MouseButtonR,
                            n => todo!("{:?}", n),
                        };
                        let value = match state {
                            ElementState::Pressed => 1.0,
                            ElementState::Released => 0.0,
                        };
                        events.push(set_input(&mut self.active_inputs, key, value));
                    }
                    DeviceEvent::MouseWheel { delta } => {}
                    e => todo!("{:?}", e),
                },
                E::UserEvent(_) => todo!(),
                E::AboutToWait => {}
                E::LoopExiting => todo!(),
                E::MemoryWarning => todo!(),
            });
        events
    }

    /// Window size
    pub fn size(&self) -> UVec2 {
        let size = self.window.inner_size();
        UVec2::new(size.width, size.height)
    }

    /// Width over height ratio.
    pub fn aspect(&self) -> f32 {
        let size = self.size();
        size.x as f32 / size.y as f32
    }

    /// Get the value of an input.
    pub fn input(&self, key: &InputKey) -> f32 {
        *self.active_inputs.get(key).unwrap_or(&0.0)
    }

    /// Reset accumulated relative mouse movements, including wheel.
    pub fn reset_mouse_relative(&mut self) {
        self.active_inputs.remove(&InputKey::MouseRelativeX);
        self.active_inputs.remove(&InputKey::MouseRelativeY);
        self.active_inputs.remove(&InputKey::MouseWheelY);
    }

    pub fn render_handles(&self) -> (WindowHandle, DisplayHandle) {
        (
            self.window.window_handle().unwrap(),
            self.window.display_handle().unwrap(),
        )
    }
}

fn set_input(map: &mut HashMap<InputKey, f32>, key: InputKey, value: f32) -> Event {
    if value == 0.0 {
        map.remove(&key);
    } else {
        map.insert(key.clone(), value);
    }
    Event::Input(Input { key, value })
}

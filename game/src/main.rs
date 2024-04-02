use std::time::Instant;

fn main() {
    //let meshes = render::Mesh::from_glb_slice(include_bytes!("/tank/games/Mechaia/a0-wheel.glb"));
    let meshes = render::Mesh::from_glb_slice(include_bytes!("/tmp/untitled.glb"));

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut vulkan = render::init_vulkan(&window);
    let mesh_set = vulkan.add_meshes(&meshes, 1024);
    let draw_set = vulkan.make_draw_set(mesh_set, render::ShaderSetHandle::PBR);

    let mut pos = glam::Vec3::new(-10.0, 0.0, 0.0);
    let mut pan @ mut tilt = 0.0;

    pos.z += 5.0;
    tilt = -0.5;

    let mut go_forward @ mut go_back @ mut go_up @ mut go_down @ mut go_left @ mut go_right = false;

    // -4.^C7015 -2.5587323 8.5509615, pan: -0.20700057, tilt: -0.50299996
    /*
    pos.x = -4.5;
    pos.y = -2.5;
    pos.z = 8.5;
    pan = -0.2;
    tilt = -0.5;
    */
    pos.z = 5.0;
    tilt = -0.5;

    let mut aspect = {
        let size = window.inner_size();
        size.width as f32 / size.height as f32
    };

    let mut t = Instant::now();
    let mut dt = 0.0;
    use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event,
            window_id: _,
        } => match event {
            WindowEvent::CloseRequested => *control_flow = winit::event_loop::ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => {
                *(match input.scancode {
                    17 => &mut go_forward,
                    31 => &mut go_back,
                    57 => &mut go_up,
                    42 => &mut go_down,
                    30 => &mut go_left,
                    32 => &mut go_right,
                    125 => return,
                    s => {
                        dbg!(s);
                        return;
                    }
                }) = input.state == ElementState::Pressed
            }
            WindowEvent::Resized(size) => {
                aspect = size.width as f32 / size.height as f32;
                vulkan.rebuild_swapchain();
            }
            _ => {}
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Motion { axis: 0, value } => pan -= value as f32 * 0.003,
            DeviceEvent::Motion { axis: 1, value } => tilt -= value as f32 * 0.003,
            _ => {}
        },
        Event::LoopDestroyed => {}
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            let rot = glam::Quat::from_euler(glam::EulerRot::YZX, tilt, pan, 0.0);
            // FIXME the fuck we need to invert for?
            pos += rot.inverse()
                * (10.0
                    * dt
                    * glam::Vec3::new(
                        f32::from(go_forward) - f32::from(go_back),
                        f32::from(go_right) - f32::from(go_left),
                        f32::from(go_up) - f32::from(go_down),
                    ));
            vulkan.draw(
                &render::Camera {
                    translation: pos,
                    rotation: rot,
                    fov: core::f32::consts::FRAC_PI_4,
                    aspect,
                    near: 1e-2,
                    far: 1e3,
                },
                draw_set,
            );

            let nt = Instant::now();
            dt = nt.duration_since(t).as_secs_f32();
            print!(
                "\rpos: {} {} {}, pan: {}, tilt: {} FPS: {}                 ",
                pos[0],
                pos[1],
                pos[2],
                pan,
                tilt,
                1.0 / dt,
            );
            t = nt;
        }
        _ => {}
    });
}

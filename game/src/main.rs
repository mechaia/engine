use std::time::Instant;

use glam::{Quat, UVec2, Vec3};

fn main() {
    let mut physics = physics3d::Physics::new();

    let phys_dt = 1.0 / 60.0;
    physics.set_time_delta(phys_dt);

    let floor_shape = physics.make_halfspace_shape(Vec3::Z);
    let floor_body = physics.make_fixed_rigid_body();
    physics.add_collider(
        floor_body,
        floor_shape,
        &physics3d::ColliderProperties {
            local_transform: physics3d::Transform {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            },
            friction: 0.2,
            //bounciness: 0.2,
            bounciness: 1.0,
        },
    );

    let ball_shape = physics.make_sphere_shape(1.0);
    let ball_body = physics.make_dynamic_rigid_body();
    physics.add_collider(
        ball_body,
        ball_shape,
        &physics3d::ColliderProperties {
            local_transform: physics3d::Transform {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            },
            friction: 0.1,
            bounciness: 1.0,
        },
    );
    physics.set_rigid_body_properties(
        ball_body,
        &physics3d::SetRigidBodyProperties {
            world_transform: Some(physics3d::Transform {
                translation: Vec3::new(0.0, -2.0, 3.0),
                rotation: Quat::IDENTITY,
            }),
            linear_velocity: Some(Vec3::X * 0.1),
            ..Default::default()
        },
    );

    //let meshes = render::Mesh::from_glb_slice(include_bytes!("/tank/games/Mechaia/a0-wheel.glb"));
    let meshes = render::Mesh::from_glb_slice(include_bytes!("/tmp/untitled.glb"));

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut render = render::Render::new(&window);

    let tex_white = render.add_texture_2d(
        UVec2::new(1, 1),
        render::TextureFormat::Rgba8Unorm,
        &mut |s| s.fill(u8::MAX),
    );

    let tex_smiley = render.add_texture_2d(
        UVec2::new(16, 16),
        render::TextureFormat::Rgba8Unorm,
        &mut |s| {
            let bitmap = [
                0b0000_0000_0000_0000,
                0b0000_0000_0000_0000,
                0b0000_1100_0011_0000,
                0b0001_1110_0111_1000,
                0b0001_1110_0111_1000,
                0b0000_1100_0011_0000,
                0b0000_0000_0000_0000,
                0b0000_0000_0000_0000,
                0b0010_0000_0000_0100,
                0b0011_0000_0000_1100,
                0b0001_1000_0001_1000,
                0b0000_1111_1111_0000,
                0b0000_0111_1110_0000,
                0b0000_0000_0000_0000,
                0b0000_0000_0000_0000,
                0b0000_0000_0000_0000,
            ];

            assert_eq!(s.len(), 16 * 16 * 4);
            for (a, b) in bitmap.iter().zip(s.chunks_mut(4 * 16)) {
                for (i, c) in b.chunks_mut(4).enumerate() {
                    c.fill(u8::MAX * (((a >> i) & 1) ^ 1) as u8);
                }
            }
        },
    );

    let mat_plain_white = render.add_pbr_material(&render::PbrMaterial {
        albedo: render::Rgb::new(1.0, 1.0, 1.0),
        roughness: 0.5,
        metallic: 0.5,
        ambient_occlusion: 1.0,
        albedo_texture: tex_white,
        roughness_texture: tex_white,
        metallic_texture: tex_white,
        ambient_occlusion_texture: tex_white,
    });
    let mat_smiley_yellow = render.add_pbr_material(&render::PbrMaterial {
        albedo: render::Rgb::new(1.0, 1.0, 0.0),
        roughness: 0.5,
        metallic: 0.5,
        ambient_occlusion: 1.0,
        albedo_texture: tex_smiley,
        roughness_texture: tex_white,
        metallic_texture: tex_white,
        ambient_occlusion_texture: tex_white,
    });

    let mesh_set = render.add_meshes(&meshes, 1024);
    let draw_set = render.make_draw_closure(mesh_set, render::ShaderSetHandle::PBR, 1024);

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

    let mut total_t = 0.0;
    let mut last_phys_t = total_t;

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
                render.rebuild_swapchain();
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
            let tr = physics.rigid_body_transform(ball_body);
            dbg!(tr);
            render.draw(
                &render::Camera {
                    translation: pos,
                    rotation: rot,
                    projection: render::CameraProjection::Perspective {
                        fov: core::f32::consts::FRAC_PI_4,
                    },
                    aspect,
                    near: 1e-2,
                    far: 1e3,
                },
                draw_set,
                &[3, 3, 1],
                &mut [
                    render::InstanceData {
                        translation: tr.translation,
                        rotation: tr.rotation,
                        material: mat_plain_white,
                        //material: mat_smiley_yellow,
                    },
                    render::InstanceData {
                        translation: Vec3::ZERO,
                        rotation: Quat::from_rotation_x(-1.0),
                        material: mat_plain_white,
                    },
                    render::InstanceData {
                        translation: Vec3::new(0.0, 4.0, 0.0),
                        rotation: Quat::from_rotation_z(total_t),
                        material: mat_plain_white,
                    },
                    render::InstanceData {
                        translation: Vec3::new(-3.0, -2.0, 0.0),
                        rotation: Quat::IDENTITY,
                        material: mat_smiley_yellow,
                    },
                    render::InstanceData {
                        translation: Vec3::new(-3.0, 2.0, 0.0),
                        rotation: Quat::from_rotation_y(2.0),
                        material: mat_plain_white,
                    },
                    render::InstanceData {
                        translation: Vec3::new(-3.0, 4.0, 0.0),
                        rotation: Quat::IDENTITY,
                        material: mat_plain_white,
                    },
                    render::InstanceData {
                        translation: Vec3::new(0.0, 0.0, 5.0),
                        rotation: Quat::IDENTITY,
                        material: mat_smiley_yellow,
                    },
                ]
                .into_iter(),
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
            total_t += dt;

            if total_t - last_phys_t >= phys_dt {
                physics.step();
                last_phys_t = total_t;
            }

            //std::thread::sleep_ms(1000);
        }
        _ => {}
    });
}

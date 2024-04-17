use mechaia::{
    math::{EulerRot, Quat, UVec2, Vec2, Vec3, Vec4},
    physics3d, render,
    window::{self, Event, InputKey},
};
use std::time::{Duration, Instant};

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
    let models = mechaia::model::gltf::from_glb_slice(include_bytes!("/tmp/untitled.glb"));

    let mut window = window::Window::new();

    let mut render = {
        let (a, b) = window.render_handles();
        render::Render::new(a, b)
    };

    let mesh_set = mechaia::render::resource::mesh::MeshSet::new(
        &mut render,
        &models
            .meshes
            .iter()
            .map(|m| mechaia::render::resource::mesh::Mesh {
                indices: &m.indices,
                vertices: m.vertices.as_slice(),
            })
            .collect::<Vec<_>>(),
    );

    let texture_set = {
        use render::resource::texture::{TextureFormat, TextureSet};
        TextureSet::builder(&mut render, TextureFormat::Rgba8Unorm, 2)
            .push(UVec2::new(1, 1), &mut |s| s.fill(u8::MAX))
            .push(UVec2::new(16, 16), &mut |s| {
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

                for (a, b) in bitmap.iter().zip(s.chunks_mut(4 * 16)) {
                    for (i, c) in b.chunks_mut(4).enumerate() {
                        c.fill(u8::MAX * (((a >> i) & 1) ^ 1) as u8);
                    }
                }
            })
            .build()
    };

    let texture_set_gui = {
        use render::resource::texture::{TextureFormat, TextureSet};
        TextureSet::builder(&mut render, TextureFormat::Rgba8Unorm, 2)
            .push(UVec2::new(1, 1), &mut |s| s.fill(u8::MAX))
            .push(UVec2::new(16, 16), &mut |s| {
                let bitmap = [
                    0b0000_0000_0000_0000,
                    0b0000_0000_0000_0000,
                    0b0000_0010_0100_0000,
                    0b0000_0001_1000_0000,
                    0b0000_0001_1000_0000,
                    0b0000_0001_1000_0000,
                    0b0010_0001_1000_0100,
                    0b0001_1111_1111_1000,
                    0b0001_1111_1111_1000,
                    0b0010_0001_1000_0100,
                    0b0000_0001_1000_0000,
                    0b0000_0001_1000_0000,
                    0b0000_0001_1000_0000,
                    0b0000_0010_0100_0000,
                    0b0000_0000_0000_0000,
                    0b0000_0000_0000_0000,
                ];

                for (a, b) in bitmap.iter().zip(s.chunks_mut(4 * 16)) {
                    for (i, c) in b.chunks_mut(4).enumerate() {
                        c.fill(u8::MAX * ((a >> i) & 1) as u8);
                    }
                }
            })
            .build()
    };

    let tex_white = 0;
    let tex_smiley = 1;

    let material_set = {
        use render::resource::material::pbr::*;
        PbrMaterialSet::builder(&mut render, 2)
            .push(&PbrMaterial {
                albedo: Vec4::new(1.0, 1.0, 1.0, 1.0),
                roughness: 0.5,
                metallic: 0.5,
                ambient_occlusion: 1.0,
                albedo_texture: tex_white,
                roughness_texture: tex_white,
                metallic_texture: tex_white,
                ambient_occlusion_texture: tex_white,
            })
            .push(&PbrMaterial {
                albedo: Vec4::new(1.0, 1.0, 0.0, 1.0),
                roughness: 0.5,
                metallic: 0.5,
                ambient_occlusion: 1.0,
                albedo_texture: tex_smiley,
                roughness_texture: tex_white,
                metallic_texture: tex_white,
                ambient_occlusion_texture: tex_white,
            })
            .build()
    };

    let mat_plain_white = 0;
    let mat_smiley_yellow = 1;

    let (mut pbr, mut gui, stage_set) = unsafe {
        let mut renderpass = render::stage::renderpass::RenderPass::builder(&mut render);
        let (pbr, compute) = render::stage::standard3d::Standard3D::new(
            &mut render,
            &mut renderpass,
            texture_set,
            material_set,
            mesh_set,
            false,
        );
        let gui = mechaia::gui::push(&mut render, &mut renderpass, 1024, texture_set_gui);
        let renderpass = renderpass.build(&mut render);

        let stage_set = render
            .add_stage_set([render::box_stage(compute), render::box_stage(renderpass)].into());

        (pbr, gui, stage_set)
    };

    let mut pos = Vec3::new(-10.0, 0.0, 0.0);
    let mut pan @ mut tilt = 0.0;

    pos.z = 5.0;
    tilt = -0.5;

    let mut total_t = 0.0;
    let mut last_phys_t = total_t;

    let start = Instant::now();

    let mut t = start;
    let mut dt = 0.0;
    loop {
        let deadline = Instant::now()
            .checked_add(Duration::from_secs_f32(1.0 / 120.0))
            .unwrap();
        window.reset_mouse_relative();
        loop {
            let events = window.wait(deadline);
            if events.is_empty() {
                break;
            }
            for evt in events {
                match evt {
                    Event::Resized(_) => render.rebuild_swapchain(),
                    Event::Input(_) => {}
                }
            }
        }

        pan -= window.input(InputKey::MouseRelativeX) * 0.003;
        tilt -= window.input(InputKey::MouseRelativeY) * 0.003;

        let f = |a, b| window.input(a) - window.input(b);
        let rot = Quat::from_euler(EulerRot::YZX, tilt, pan, 0.0);
        // FIXME the fuck we need to invert for?
        pos += rot.inverse()
            * (10.0
                * dt
                * Vec3::new(
                    f(InputKey::Z, InputKey::S),
                    f(InputKey::D, InputKey::Q),
                    f(InputKey::Space, InputKey::LCtrl),
                ));
        let tr = physics.rigid_body_transform(ball_body);

        let mut f = |index| unsafe {
            use render::stage::standard3d::{Instance, Transform};
            let data = [
                Transform {
                    translation: tr.translation,
                    rotation: tr.rotation,
                },
                Transform {
                    translation: Vec3::ZERO,
                    rotation: Quat::from_rotation_x(-1.0),
                },
                Transform {
                    translation: Vec3::new(0.0, 4.0, 0.0),
                    rotation: Quat::from_rotation_z(total_t),
                },
                Transform {
                    translation: Vec3::new(-3.0, -2.0, 0.0),
                    rotation: Quat::IDENTITY,
                },
                Transform {
                    translation: Vec3::new(-3.0, 2.0, 0.0),
                    rotation: Quat::from_rotation_y(2.0),
                },
                Transform {
                    translation: Vec3::new(-3.0, 4.0, 0.0),
                    rotation: Quat::IDENTITY,
                },
                Transform {
                    translation: Vec3::new(0.0, 0.0, 5.0),
                    rotation: Quat::IDENTITY,
                },
                Transform {
                    translation: Vec3::new(0.0, 0.0, 3.0),
                    rotation: Quat::IDENTITY,
                },
            ];
            pbr.set_transform_data(index, &mut data.into_iter());
            let data = [
                Instance {
                    transforms_offset: 0,
                    material: mat_smiley_yellow,
                },
                Instance {
                    transforms_offset: 1,
                    material: mat_plain_white,
                },
                Instance {
                    transforms_offset: 2,
                    material: mat_plain_white,
                },
                Instance {
                    transforms_offset: 3,
                    material: mat_smiley_yellow,
                },
                Instance {
                    transforms_offset: 4,
                    material: mat_plain_white,
                },
                Instance {
                    transforms_offset: 5,
                    material: mat_plain_white,
                },
                Instance {
                    transforms_offset: 6,
                    material: mat_smiley_yellow,
                },
            ];
            pbr.set_instance_data(index, &[3, 3, 1], &mut data.into_iter());

            gui.draw(
                index,
                &mut ((0..10)
                    .flat_map(|y| (0..10).map(move |x| (x, y)))
                    .map(|(x, y)| mechaia::gui::Instance {
                        half_extents: Vec2::ONE / 32.0,
                        position: Vec2::NEG_ONE + Vec2::new(x as f32 / 10.0, y as f32 / 10.0) * 2.0,
                        rotation: t.duration_since(start).as_secs_f32(),
                        uv_start: Vec2::ZERO,
                        uv_end: Vec2::ONE,
                        texture: (x ^ y) & 1,
                    }))
                .into_iter() as &mut dyn Iterator<Item = _>,
            );
        };

        render.draw(
            &render::Camera {
                translation: pos,
                rotation: rot,
                projection: render::CameraProjection::Perspective {
                    fov: core::f32::consts::FRAC_PI_4,
                },
                aspect: window.aspect(),
                near: 1e-2,
                far: 1e3,
            },
            stage_set,
            &mut f,
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
    }
}

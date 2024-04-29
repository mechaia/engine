pub mod font;
pub mod immediate;

use ash::vk;
use core::{ffi::CStr, mem};
use glam::{U16Vec2, Vec2, Vec4};
use render::{
    resource::{texture::TextureView, Shared},
    stage::renderpass::{
        RenderPassBuilder, RenderSubpass, RenderSubpassBuilder, SubpassAttachmentReferences,
    },
    Render,
};
use std::{
    ptr::NonNull,
    sync::{Arc, Mutex, MutexGuard},
};

const ENTRY_POINT: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

pub struct Gui {
    shared: Data,
}

pub struct Configuration {
    pub max_instances: u32,
    pub max_textures: u32,
}

pub struct Draw<'a> {
    viewport: U16Vec2,
    shared: MutexGuard<'a, SharedData>,
    index: usize,
    instance_data_ptr: *mut InstanceData,
    instance_data_end: *mut InstanceData,
}

pub struct Instance {
    pub position: U16Vec2,
    pub size: U16Vec2,
    pub rotation: f32,
    pub uv_start: Vec2,
    pub uv_end: Vec2,
    pub texture: u32,
    pub color: Vec4,
}

type Data = Arc<Mutex<SharedData>>;

struct SharedData {
    max_instances: u32,
    data: Box<[SubpassData]>,
    viewport: U16Vec2,
    max_textures: u32,
    textures: util::Arena<Shared<TextureView>>,
}

struct Subpass {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    shared: Data,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
}

struct SubpassData {
    instance_data: render::VmaBuffer,
    instance_data_ptr: NonNull<InstanceData>,
    draw_parameters: render::VmaBuffer,
    draw_parameters_ptr: NonNull<vk::DrawIndirectCommand>,
    set: vk::DescriptorSet,
}

struct SubpassBuilder {
    shared: Data,
    max_textures: u32,
}

#[repr(C)]
struct InstanceData {
    position: U16Vec2,
    size: U16Vec2,
    rotation: f32,
    texture_index: u32,
    uv_start: Vec2,
    uv_end: Vec2,
    color: [f32; 4],
}

impl Gui {
    pub fn draw<'a>(&'a mut self, index: usize) -> Draw<'a> {
        let shared = self.shared.lock().unwrap();
        let data = &shared.data[index];
        unsafe {
            let ptr = data.instance_data_ptr.as_ptr();
            Draw {
                viewport: shared.viewport,
                instance_data_ptr: ptr,
                instance_data_end: ptr.add(usize::try_from(shared.max_instances).unwrap()),
                shared,
                index,
            }
        }
    }

    pub fn add_texture(&mut self, render: &mut Render, texture: Shared<TextureView>) {
        let sampler = render.dev_mut().nearest_sampler();
        let infos = [texture.bind_info(sampler)];

        let mut shared = self.shared.lock().unwrap();
        let h = shared.textures.insert(texture);
        assert!(h.as_u32() < shared.max_textures);

        let writes = shared
            .data
            .iter()
            .map(|d| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(d.set)
                    .dst_binding(0)
                    .dst_array_element(h.as_u32())
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&infos)
            })
            .map(|w| w.build())
            .collect::<Vec<_>>();

        unsafe { render.dev_mut().update_descriptor_sets(&writes, &[]) };
    }
}

impl<'a> Draw<'a> {
    pub fn viewport(&self) -> U16Vec2 {
        self.viewport
    }

    pub fn push(&mut self, instance: &Instance) {
        assert!(
            self.instance_data_ptr < self.instance_data_end,
            "out of bounds"
        );
        unsafe {
            self.instance_data_ptr.write(InstanceData {
                position: instance.position,
                size: instance.size,
                rotation: instance.rotation,
                uv_start: instance.uv_start,
                uv_end: instance.uv_end,
                texture_index: instance.texture,
                color: instance.color.to_array(),
            });
            self.instance_data_ptr = self.instance_data_ptr.add(1);
        }
    }
}

impl Drop for Draw<'_> {
    fn drop(&mut self) {
        let data = &self.shared.data[self.index];
        unsafe {
            let instance_count = self
                .instance_data_ptr
                .offset_from(data.instance_data_ptr.as_ptr())
                .try_into()
                .unwrap();
            data.draw_parameters_ptr
                .as_ptr()
                .write(vk::DrawIndirectCommand {
                    vertex_count: 4,
                    instance_count,
                    first_vertex: 0,
                    first_instance: 0,
                });
        }
    }
}

#[must_use]
pub fn push(
    render: &mut Render,
    render_pass: &mut RenderPassBuilder,
    config: &Configuration,
) -> Gui {
    unsafe {
        let shared = Arc::new(Mutex::new(SharedData {
            data: [].into(),
            max_instances: config.max_instances,
            viewport: U16Vec2::ZERO,
            textures: Default::default(),
            max_textures: config.max_textures,
        }));

        let subpass = render_pass.push(
            SubpassBuilder {
                shared: shared.clone(),
                max_textures: config.max_textures,
            },
            SubpassAttachmentReferences {
                color: Box::new([vk::AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }]),
                depth_stencil: None,
            },
        );
        assert!(subpass > 0, "dependency on a previous subpass!");
        render_pass.add_dependency(vk::SubpassDependency {
            dependency_flags: vk::DependencyFlags::empty(),
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            src_subpass: subpass - 1,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_subpass: subpass,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        });

        Gui { shared }
    }
}

unsafe impl RenderSubpassBuilder for SubpassBuilder {
    unsafe fn build(
        self: Box<Self>,
        dev: &render::Dev,
        render_pass: vk::RenderPass,
        subpass: u32,
    ) -> Box<dyn render::stage::renderpass::RenderSubpass> {
        let vertex_shader = make_shader(
            dev,
            vk_shader_macros::include_glsl!("shader/quad.glsl.vert", kind: vert),
        );
        let fragment_shader = make_shader(
            dev,
            vk_shader_macros::include_glsl!("shader/quad.glsl.frag", kind: frag),
        );

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(ENTRY_POINT),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(ENTRY_POINT),
        ]
        .map(|x| x.build());

        let attr_descrs = {
            let mut loc = 0;
            let mut f = |offset: u32, format| {
                loc += 1;
                vk::VertexInputAttributeDescription {
                    location: loc - 1,
                    binding: 0,
                    format,
                    offset: offset * 4,
                }
            };
            [
                f(0, vk::Format::R16G16_USCALED),
                f(1, vk::Format::R16G16_USCALED),
                f(2, vk::Format::R32_SFLOAT),
                f(3, vk::Format::R32_UINT),
                f(4, vk::Format::R32G32_SFLOAT),
                f(6, vk::Format::R32G32_SFLOAT),
                f(8, vk::Format::R32G32B32A32_SFLOAT),
            ]
        };
        let binding_descrs = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 12 * 4,
            input_rate: vk::VertexInputRate::INSTANCE,
        }];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&attr_descrs)
            .vertex_binding_descriptions(&binding_descrs);
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        // set to default as we'll set it in the commands instead.
        let viewports = [vk::Viewport::default()];
        let scissors = [vk::Rect2D::default()];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let colorblend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colorblend_attachments);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(self.max_textures)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()];
            let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            dev.create_descriptor_set_layout(&info, None).unwrap()
        };

        let layout = {
            let layouts = [descriptor_set_layout];
            let push_constant_ranges = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: 8,
            }];
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_ranges);
            dev.create_pipeline_layout(&info, None).unwrap()
        };

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .color_blend_state(&colorblend_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(subpass)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .build()];
        let pipeline = dev
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0];

        dev.destroy_shader_module(fragment_shader, None);
        dev.destroy_shader_module(vertex_shader, None);

        let pool = make_descriptor_pool(dev, self.max_textures, 5 /* FIXME */);

        Box::new(Subpass {
            pipeline,
            layout,
            shared: self.shared,
            descriptor_set_layout,
            pool,
        })
    }
}

unsafe impl RenderSubpass for Subpass {
    unsafe fn record_commands(&self, dev: &render::Dev, args: &render::StageArgs) {
        let sh = self.shared.lock().unwrap();
        let data = &sh.data[args.index];

        dev.cmd_bind_pipeline(args.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        dev.cmd_bind_descriptor_sets(
            args.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[data.set],
            &[],
        );
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: args.viewport.width as f32,
            height: args.viewport.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: args.viewport,
        }];

        let mut c = [0; 8];
        c[..4].copy_from_slice(&(args.viewport.width as f32).to_le_bytes());
        c[4..].copy_from_slice(&(args.viewport.height as f32).to_le_bytes());

        dev.cmd_push_constants(args.cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, &c);

        dev.cmd_set_viewport(args.cmd, 0, &viewports);
        dev.cmd_set_scissor(args.cmd, 0, &scissors);
        dev.cmd_bind_vertex_buffers(args.cmd, 0, &[data.instance_data.0], &[0]);
        dev.cmd_draw_indirect(
            args.cmd,
            data.draw_parameters.0,
            0,
            1,
            mem::size_of::<vk::DrawIndirectCommand>() as u32,
        );
    }

    unsafe fn rebuild_swapchain(&mut self, dev: &mut render::Dev, swapchain: &render::SwapChain) {
        let mut sh = self.shared.lock().unwrap();
        let mut data = mem::take(&mut sh.data).into_vec();

        let new_len = swapchain.image_count();
        if new_len > data.len() {
            let mut sets = {
                let layouts = (data.len()..new_len)
                    .map(|_| self.descriptor_set_layout)
                    .collect::<Vec<_>>();
                let info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts);
                dev.allocate_descriptor_sets(&info).unwrap()
            };

            let old_len = data.len();

            data.resize_with(new_len, || {
                let mut instance_data = dev.allocate_buffer(
                    4 * 8 * u64::from(sh.max_instances),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                );
                let mut draw_parameters = dev.allocate_buffer(
                    4 + 1 * mem::size_of::<vk::DrawIndirectCommand>() as u64,
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                );
                let instance_data_ptr = dev.map_buffer(&mut instance_data).cast::<InstanceData>();
                let draw_parameters_ptr = dev
                    .map_buffer(&mut draw_parameters)
                    .cast::<vk::DrawIndirectCommand>();
                draw_parameters_ptr.as_ptr().write(vk::DrawIndirectCommand {
                    vertex_count: 0,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                });

                SubpassData {
                    instance_data,
                    instance_data_ptr,
                    draw_parameters,
                    draw_parameters_ptr,
                    set: sets.pop().unwrap(),
                }
            });

            let sampler = dev.nearest_sampler();
            let infos = sh
                .textures
                .values()
                .map(|t| [t.bind_info(sampler)])
                .collect::<Vec<_>>();
            let writes = data[old_len..]
                .iter()
                .flat_map(|d| {
                    sh.textures
                        .keys()
                        .zip(&infos)
                        .map(|(h, infos)| {
                            vk::WriteDescriptorSet::builder()
                                .dst_set(d.set)
                                .dst_binding(0)
                                .dst_array_element(h.as_u32())
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(infos)
                        })
                        .map(|w| w.build())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            unsafe { dev.update_descriptor_sets(&writes, &[]) };
        } else {
            while data.len() > new_len {
                data.pop().unwrap().drop_with(dev, self.pool);
            }
        }

        sh.data = data.into();

        let ext = swapchain.extent();
        sh.viewport.x = ext.width.try_into().unwrap();
        sh.viewport.y = ext.height.try_into().unwrap();
    }
}

impl SubpassData {
    fn drop_with(mut self, dev: &mut render::Dev, pool: vk::DescriptorPool) {
        unsafe {
            dev.free_descriptor_sets(pool, &[self.set]).unwrap();
            dev.unmap_buffer(&mut self.instance_data);
            dev.free_buffer(self.instance_data);
            dev.unmap_buffer(&mut self.draw_parameters);
            dev.free_buffer(self.draw_parameters);
        }
    }
}

fn make_descriptor_pool(
    dev: &render::Dev,
    max_texture_count: u32,
    image_count: u32,
) -> vk::DescriptorPool {
    let sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: image_count * max_texture_count,
    }];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&sizes)
        .max_sets(image_count);
    unsafe { dev.create_descriptor_pool(&info, None).unwrap() }
}

fn make_shader(dev: &render::Dev, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    unsafe { dev.create_shader_module(&info, None).unwrap() }
}

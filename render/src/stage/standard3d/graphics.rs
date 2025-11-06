use crate::{
    resource::mesh::MeshSet,
    stage::renderpass::{RenderPassBuilder, RenderSubpassBuilder, SubpassAttachmentReferences},
    DropWith, Render, VmaBuffer,
};
use ash::vk;
use core::{mem, ptr::NonNull};
use glam::{UVec2, Vec3};

struct GraphicsSubpass {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    shared: super::Data,
}

struct GraphicsSubpassBuilder {
    shared: super::Data,
    transparent: bool,
}

/// Per-stage data
pub(super) struct Data {
    /// Projected transforms
    ///
    /// - transforms (mat4)
    ///
    /// Set by compute shader, so device-local.
    pub(super) data_transforms: VmaBuffer,
    /// Instance data
    ///
    /// - instance data
    ///
    /// Set by host.
    pub(super) data_instances: VmaBuffer,
    pub(super) data_instances_ptr: NonNull<InstanceData>,
    pub(super) has_empty_draw_issue: bool,
}

/// Per-set data
pub(super) struct SetData {
    per_image: Box<[SetDataPerImage]>,
}

/// Per-set per-image data
struct SetDataPerImage {
    /// Draw parameters
    ///
    /// - parameters count
    /// - parameters array
    ///
    /// (currently) Set by host, so host-visible.
    pub(super) parameters: VmaBuffer,
    /// Pointer to mapped parameters.
    pub(super) parameters_ptr: NonNull<u8>,
}

#[repr(C)]
pub(super) struct InstanceData {
    transforms_offset: u32,
    material_index: u32,
}

impl Data {
    pub unsafe fn set_instance_data(
        &mut self,
        index: usize,
        set_data: &util::Arena<super::SetData>,
        mut instance_counts: &[u32],
        instance_data: &mut dyn Iterator<Item = super::Instance>,
    ) {
        let count = instance_counts.iter().sum::<u32>();

        let mut p = self.data_instances_ptr.as_ptr();
        for d in instance_data.take(usize::try_from(count).unwrap()) {
            p.write(InstanceData {
                transforms_offset: d.transforms_offset,
                material_index: d.material,
            });
            p = p.add(1);
        }

        let mut first_instance = 0;

        for set in set_data.values() {
            let p = set.graphics.per_image[index]
                .parameters_ptr
                .cast::<u32>()
                .as_ptr();
            let ic;
            (ic, instance_counts) = instance_counts.split_at(set.mesh_set.len());
            let draw_count = ic.iter().filter(|&&n| n > 0).count() as u32;

            if draw_count == 0 && self.has_empty_draw_issue {
                let p = set.graphics.per_image[index]
                    .parameters_ptr
                    .cast::<u32>()
                    .as_ptr();
                p.write(1);
                p.add(1)
                    .cast::<vk::DrawIndexedIndirectCommand>()
                    .write(Default::default());
                continue;
            }

            p.write(draw_count as u32);

            let mut p = p.add(1).cast::<vk::DrawIndexedIndirectCommand>();
            for (i, &instance_count) in ic.iter().enumerate() {
                if instance_count == 0 {
                    continue;
                }
                let mesh = set.mesh_set.mesh(i);
                p.write(vk::DrawIndexedIndirectCommand {
                    first_instance,
                    instance_count,
                    index_count: mesh.index_count,
                    first_index: mesh.index_offset,
                    vertex_offset: mesh.vertex_offset as i32,
                });
                first_instance += instance_count;
                p = p.add(1);
            }
        }
    }
}

unsafe impl DropWith for SetData {
    fn drop_with(self, dev: &mut crate::Dev) {
        for mut pi in self.per_image.into_vec().drain(..) {
            unsafe {
                dev.unmap_buffer(&mut pi.parameters);
                dev.free_buffer(pi.parameters);
            }
        }
    }
}

pub unsafe fn push(
    render: &mut Render,
    render_pass: &mut RenderPassBuilder,
    shared: super::Data,
    transparent: bool,
) {
    let subpass = render_pass.push(
        GraphicsSubpassBuilder {
            shared,
            transparent,
        },
        SubpassAttachmentReferences {
            color: Box::new([vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }]),
            depth_stencil: Some(vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
        },
    );
    let src_subpass = if subpass == 0 {
        render_pass.add_attachment(vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: render.swapchain.format(),
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            samples: vk::SampleCountFlags::TYPE_1,
        });
        render_pass.add_attachment(vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk::Format::D32_SFLOAT,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
        });
        vk::SUBPASS_EXTERNAL
    } else {
        subpass - 1
    };
    render_pass.add_dependency(vk::SubpassDependency {
        dependency_flags: vk::DependencyFlags::empty(),
        src_access_mask: vk::AccessFlags::NONE,
        src_subpass,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        dst_subpass: subpass,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    });
}

unsafe impl RenderSubpassBuilder for GraphicsSubpassBuilder {
    unsafe fn build(
        self: Box<Self>,
        dev: &crate::Dev,
        render_pass: vk::RenderPass,
        subpass: u32,
    ) -> Box<dyn crate::stage::renderpass::RenderSubpass> {
        let vertex_shader = super::make_shader(
            dev,
            vk_shader_macros::include_glsl!("shader/pbr.vert.glsl", kind: vert),
        );
        let fragment_shader = super::make_shader(
            dev,
            vk_shader_macros::include_glsl!("shader/pbr.frag.glsl", kind: frag),
        );

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(super::ENTRY_POINT),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(super::ENTRY_POINT),
        ]
        .map(|x| x.build());

        let mut location = 0;
        let mut f = |binding, format, offset| {
            location += 1;
            vk::VertexInputAttributeDescription {
                binding,
                location: location - 1,
                format,
                offset,
            }
        };
        let f_v = |binding, stride| vk::VertexInputBindingDescription {
            binding,
            stride,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let f_i = |binding, stride| vk::VertexInputBindingDescription {
            binding,
            stride,
            input_rate: vk::VertexInputRate::INSTANCE,
        };
        let attr_descrs = [
            // vertex
            f(0, vk::Format::R32G32B32_SFLOAT, 0),
            f(1, vk::Format::R32G32B32_SFLOAT, 0),
            f(2, vk::Format::R32G32_SFLOAT, 0),
            f(3, vk::Format::R16G16B16A16_UINT, 0),
            f(4, vk::Format::R32G32B32A32_SFLOAT, 0),
            // instance
            f(5, vk::Format::R32_UINT, 0),
            f(5, vk::Format::R32_UINT, 4),
        ];
        let binding_descrs = [
            f_v(0, 12),
            f_v(1, 12),
            f_v(2, 8),
            f_v(3, 8),
            f_v(4, 16),
            f_i(5, 8),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&attr_descrs)
            .vertex_binding_descriptions(&binding_descrs);
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        // set to default as we'll set it in the commands instead.
        let viewports = [vk::Viewport::default()];
        let scissors = [vk::Rect2D::default()];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(self.transparent)
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
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let layout = {
            let shared = self.shared.lock().unwrap();
            let layouts = [shared.descriptors.layout];
            let push_constant_ranges = [
                // vec2 inv_viewport
                // float viewport_y_over_x
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: 4 * 3,
                },
            ];
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

        Box::new(GraphicsSubpass {
            pipeline,
            layout,
            shared: self.shared,
        })
    }
}

unsafe impl crate::stage::renderpass::RenderSubpass for GraphicsSubpass {
    unsafe fn record_commands(&self, dev: &crate::Dev, args: &crate::StageArgs) {
        let shared = self.shared.lock().unwrap();

        dev.cmd_bind_pipeline(args.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        let viewport = UVec2::new(args.viewport.width, args.viewport.height).as_vec2();
        let inv_viewport = {
            let inv_vp = 1.0 / viewport;
            Vec3::new(inv_vp.x, inv_vp.y, viewport.y * inv_vp.x)
        };
        dev.cmd_push_constants(
            args.cmd,
            self.layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            crate::f32_to_bytes(&inv_viewport.to_array()),
        );
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: viewport.x,
            height: viewport.y,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: args.viewport,
        }];
        dev.cmd_set_viewport(args.cmd, 0, &viewports);
        dev.cmd_set_scissor(args.cmd, 0, &scissors);

        dev.cmd_bind_descriptor_sets(
            args.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[shared.descriptors.sets[args.index]],
            &[],
        );

        let pi_data = &shared.per_image[args.index];

        for set in shared.set_data.values() {
            dev.cmd_bind_vertex_buffers(
                args.cmd,
                0,
                &[
                    set.mesh_set.vertex_data.0,
                    set.mesh_set.vertex_data.0,
                    set.mesh_set.vertex_data.0,
                    set.mesh_set.vertex_data.0,
                    set.mesh_set.vertex_data.0,
                    pi_data.graphics.data_instances.0,
                ],
                &[
                    set.mesh_set.positions_offset,
                    set.mesh_set.normals_offset,
                    set.mesh_set.uvs_offset,
                    set.mesh_set.joints_offset,
                    set.mesh_set.weights_offset,
                    0,
                ],
            );
            dev.cmd_bind_index_buffer(
                args.cmd,
                set.mesh_set.index_data.0,
                0,
                vk::IndexType::UINT32,
            );

            dev.cmd_draw_indexed_indirect_count(
                args.cmd,
                set.graphics.per_image[args.index].parameters.0,
                4,
                set.graphics.per_image[args.index].parameters.0,
                0,
                set.mesh_set.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
    }

    unsafe fn rebuild_swapchain(
        &mut self,
        _dev: &mut crate::Dev,
        _swapchain: &crate::swapchain::SwapChain,
    ) {
    }
}

impl SetData {
    pub fn new(render: &mut crate::Render, mesh_set: &MeshSet) -> Self {
        let image_count = unsafe { render.swapchain.image_count() };
        let per_image = (0..image_count)
            .map(|_index| {
                let mut parameters = render.dev.allocate_buffer(
                    4 + super::buffer_size::<vk::DrawIndexedIndirectCommand>(
                        u32::try_from(mesh_set.len()).unwrap(),
                    ),
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                );
                let parameters_ptr = render.dev.map_buffer(&mut parameters);

                SetDataPerImage {
                    parameters,
                    parameters_ptr,
                }
            })
            .collect();

        Self { per_image }
    }
}

use ash::vk::{self, PipelineLayoutCreateFlags};
use core::ffi::CStr;
use vk_shader_macros::include_glsl;

const ENTRY_POINT: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

pub struct DrawFunction {
    pub(super) compute_pipeline: ComputePipeline,
    pub(super) graphics_pipeline: GraphicsPipeline,
}

pub(super) struct ComputePipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
}

pub(super) struct GraphicsPipeline {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
}

impl DrawFunction {
    pub unsafe fn new(
        dev: &ash::Device,
        format: vk::Format,
        render_pass: vk::RenderPass,
        texture_set_layout: vk::DescriptorSetLayout,
    ) -> Self {
        let compute_pipeline = make_compute_pipeline(dev);
        let graphics_pipeline = make_graphics_pipeline(dev, render_pass, texture_set_layout);
        Self {
            compute_pipeline,
            graphics_pipeline,
        }
    }
}

unsafe fn make_shader(dev: &ash::Device, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    dev.create_shader_module(&info, None).unwrap()
}

unsafe fn make_compute_pipeline(dev: &ash::Device) -> ComputePipeline {
    let instance_shader = make_shader(dev, include_glsl!("instance.glsl", kind: comp));
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(instance_shader)
        .name(ENTRY_POINT)
        .build();

    let descriptor_set_layout = {
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::empty())
            .bindings(&bindings);
        dev.create_descriptor_set_layout(&info, None).unwrap()
    };
    let layout = {
        let layouts = [descriptor_set_layout];
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(PipelineLayoutCreateFlags::empty())
            .set_layouts(&layouts);
        dev.create_pipeline_layout(&info, None).unwrap()
    };

    let info = Vec::from([vk::ComputePipelineCreateInfo::builder()
        .flags(vk::PipelineCreateFlags::empty())
        .stage(stage)
        .layout(layout)
        .build()]);
    let pipeline = dev
        .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        .unwrap()[0];

    dev.destroy_shader_module(instance_shader, None);

    ComputePipeline {
        pipeline,
        layout,
        descriptor_set_layout,
    }
}

/// some experimental stuff for nicer pipeline descriptions
mod description {
    pub struct MakeGraphicsPipeline<'a> {
        pub vertex_shader: &'a [u32],
        pub fragment_shader: &'a [u32],
        pub inputs: MakeGraphicsPipelineInputs<'a>,
        pub bindings: &'a [MakeGraphicsPipelineBinding<'a>],
    }

    pub struct MakeGraphicsPipelineInputs<'a> {
        pub vertex: &'a [&'a [Type]],
        pub instance: &'a [&'a [Type]],
    }

    pub struct MakeGraphicsPipelineBinding<'a> {
        pub data: MakeGraphicsPipelineBindingData<'a>,
        pub count: u32,
        pub in_vertex: bool,
        pub in_fragment: bool,
    }

    pub enum MakeGraphicsPipelineBindingData<'a> {
        UniformBuffer(&'a [Type]),
        StorageBuffer(&'a [Type]),
        Image,
    }

    pub enum Type {
        F32,
        F32_2,
        F32_3,
        F32_4,
        F32_4_4,
        U32,
    }

    impl Type {
        pub fn format(&self) -> ash::vk::Format {
            use ash::vk::Format;
            match self {
                Self::F32 => Format::R32_SFLOAT,
                Self::F32_2 => Format::R32G32_SFLOAT,
                Self::F32_3 => Format::R32G32B32_SFLOAT,
                Self::F32_4 | Self::F32_4_4 => Format::R32G32B32A32_SFLOAT,
                Self::U32 => Format::R32_UINT,
            }
        }

        pub fn format_byte_size(&self) -> u32 {
            match self {
                Self::F32 | Self::U32 => 4,
                Self::F32_2 => 8,
                Self::F32_3 => 12,
                Self::F32_4 | Self::F32_4_4 => 16,
            }
        }

        pub fn count(&self) -> usize {
            match self {
                Self::F32 | Self::F32_2 | Self::F32_3 | Self::F32_4 | Self::U32 => 1,
                Self::F32_4_4 => 4,
            }
        }
    }
}

unsafe fn make_graphics_pipeline(
    dev: &ash::Device,
    render_pass: vk::RenderPass,
    texture_set_layout: vk::DescriptorSetLayout,
) -> GraphicsPipeline {
    use description::*;

    let args = if false {
        MakeGraphicsPipeline {
            vertex_shader: include_glsl!("vertex.glsl", kind: vert),
            fragment_shader: include_glsl!("fragment.glsl", kind: frag),
            inputs: MakeGraphicsPipelineInputs {
                // position, normal, uv
                vertex: &[&[Type::F32_3], &[Type::F32_3], &[Type::F32_2]],
                // model to NDC
                instance: &[&[Type::F32_4_4]],
            },
            bindings: &[MakeGraphicsPipelineBinding {
                data: MakeGraphicsPipelineBindingData::UniformBuffer(&[Type::F32_4_4]),
                count: 1,
                in_vertex: false,
                in_fragment: true,
            }],
        }
    } else {
        MakeGraphicsPipeline {
            vertex_shader: include_glsl!("shader/pbr.vert.glsl", kind: vert),
            fragment_shader: include_glsl!("shader/pbr.frag.glsl", kind: frag),
            inputs: MakeGraphicsPipelineInputs {
                // position, normal, uv
                vertex: &[&[Type::F32_3], &[Type::F32_3], &[Type::F32_2]],
                // material index, model to NDC,
                // FIXME do'nt pad material index
                instance: &[
                    &[Type::F32_3, Type::F32, Type::F32_3, Type::U32],
                    &[Type::F32_4_4],
                ],
            },
            bindings: &[
                // transform
                MakeGraphicsPipelineBinding {
                    data: MakeGraphicsPipelineBindingData::UniformBuffer(&[Type::F32_4_4]),
                    count: 1,
                    in_vertex: true,
                    in_fragment: false,
                },
                /*
                // material
                MakeGraphicsPipelineBinding {
                    data: MakeGraphicsPipelineBindingData::StorageBuffer(&[
                        // uv
                        Type::F32_2,
                        Type::F32_2,
                        // albedo
                        Type::F32_3,
                        Type::U32,
                        // roughness
                        Type::F32,
                        Type::U32,
                        // metallic
                        Type::F32,
                        Type::U32,
                        // ambient occlusion
                        Type::F32,
                        Type::U32,
                    ]),
                    count: 1,
                    in_vertex: false,
                    in_fragment: true,
                },
                */
                // directional light
                MakeGraphicsPipelineBinding {
                    data: MakeGraphicsPipelineBindingData::StorageBuffer(&[
                        Type::F32_3,
                        Type::F32_3,
                    ]),
                    count: 1,
                    in_vertex: false,
                    in_fragment: true,
                },
            ],
        }
    };

    let vertex_shader = make_shader(dev, args.vertex_shader);
    let fragment_shader = make_shader(dev, args.fragment_shader);

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(ENTRY_POINT)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(ENTRY_POINT)
            .build(),
    ];

    let mut attr_descrs = Vec::new();
    let mut binding_descrs = Vec::new();
    let mut binding @ mut location = 0;
    for (input_rate, list) in [
        (vk::VertexInputRate::VERTEX, args.inputs.vertex),
        (vk::VertexInputRate::INSTANCE, args.inputs.instance),
    ] {
        for v_inputs in list {
            let mut offset = 0;
            for v_input in v_inputs.iter() {
                for format in core::iter::repeat(v_input.format()).take(v_input.count()) {
                    attr_descrs.push(vk::VertexInputAttributeDescription {
                        binding,
                        location,
                        format,
                        offset,
                    });
                    location += 1;
                    offset += v_input.format_byte_size();
                }
            }
            binding_descrs.push(vk::VertexInputBindingDescription {
                binding,
                stride: offset,
                input_rate,
            });
            binding += 1;
        }
    }

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
        .blend_enable(false)
        /*
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        */
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build()];
    let colorblend_info =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colorblend_attachments);

    /*
    let sampler_rgba8 = unsafe {
        let info = vk::SamplerCreateInfo::builder()
                            .mag_filter(vk::Filter::NEAREST)
                            .min_filter(vk::Filter::NEAREST)
                            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                            .address_mode_u(vk::SamplerAddressMode::REPEAT)
                            .address_mode_v(vk::SamplerAddressMode::REPEAT)
                            .address_mode_w(vk::SamplerAddressMode::REPEAT)
                            .mip_lod_bias(0.0)
                            .anisotropy_enable(false)
                            .compare_enable(false)
                            .compare_op(vk::CompareOp::NEVER)
                            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
                            .unnormalized_coordinates(false);
        dev.create_sampler(&info, None).unwrap()
    };
    let samplers = [sampler_rgba8];
    */

    let descriptor_set_layout = {
        let mut bindings = Vec::new();
        let mut binding = 0;
        for b in args.bindings.iter() {
            let mut stage_flags = vk::ShaderStageFlags::empty();
            if b.in_vertex {
                stage_flags |= vk::ShaderStageFlags::VERTEX
            }
            if b.in_fragment {
                stage_flags |= vk::ShaderStageFlags::FRAGMENT
            }
            let ty = match b.data {
                MakeGraphicsPipelineBindingData::UniformBuffer(_) => {
                    vk::DescriptorType::UNIFORM_BUFFER
                }
                MakeGraphicsPipelineBindingData::StorageBuffer(_) => {
                    vk::DescriptorType::STORAGE_BUFFER
                }
                MakeGraphicsPipelineBindingData::Image => {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                }
            };
            bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    // FIXME the validation layers segfault if you set the binding to something
                    // like .binding(4) here and in the shader,
                    // then do update_descriptor_sets on binding 0
                    //
                    // We should report it
                    .binding(binding)
                    .descriptor_type(ty)
                    .descriptor_count(b.count)
                    .stage_flags(stage_flags)
                    .build(),
            );
            binding += 1;
        }
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        dev.create_descriptor_set_layout(&info, None).unwrap()
    };

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let layout = {
        let layouts = [descriptor_set_layout, texture_set_layout];
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
        .subpass(0)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .build()];
    let pipeline = dev
        .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        .unwrap()[0];

    dev.destroy_shader_module(fragment_shader, None);
    dev.destroy_shader_module(vertex_shader, None);

    GraphicsPipeline {
        pipeline,
        layout,
        descriptor_set_layout,
    }
}

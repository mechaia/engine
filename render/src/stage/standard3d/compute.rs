use crate::{resource::mesh::MeshSet, VmaBuffer};
use ash::vk;
use core::ptr::NonNull;

pub struct ComputeStage {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    shared: super::Shared,
}

pub(super) struct Data {
    /// Instance transforms.
    ///
    /// - transform data
    ///
    /// Set by host, so host-visible.
    pub(super) data: VmaBuffer,
    /// Pointer to mapped transform data.
    pub(super) data_ptr: NonNull<TransformData>,
    /// Dispatch parameters
    ///
    /// - parameters
    ///
    /// Set by host, so host-visible.
    pub(super) parameters: VmaBuffer,
    /// Pointer to mapped parameters.
    pub(super) parameters_ptr: NonNull<vk::DispatchIndirectCommand>,
    /// - camera mat4 (uniform)
    /// - compute_data (storage)
    /// - graphics_data (storage)
    pub(super) descriptor_set: vk::DescriptorSet,
}

#[repr(C)]
pub(super) struct TransformData {
    rot: [f32; 4],
    pos: [f32; 3],
    scale: f32,
}

impl Data {
    pub unsafe fn set_transform_data(
        &mut self,
        meshes: &MeshSet,
        transform_data: &mut dyn Iterator<Item = super::Transform>,
    ) {
        let mut p = self.data_ptr.as_ptr();
        for d in transform_data {
            p.write(TransformData {
                rot: d.rotation.to_array(),
                pos: d.translation.to_array(),
                scale: d.scale,
            });
            p = p.add(1);
        }

        let count = p.offset_from(self.data_ptr.as_ptr()).try_into().unwrap();
        self.parameters_ptr
            .as_ptr()
            .write(vk::DispatchIndirectCommand {
                x: count,
                y: 1,
                z: 1,
            });
    }
}

pub unsafe fn new(shared: super::Shared, dev: &ash::Device) -> ComputeStage {
    let instance_shader = super::make_shader(
        dev,
        vk_shader_macros::include_glsl!("instance.glsl", kind: comp),
    );
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(instance_shader)
        .name(super::ENTRY_POINT)
        .build();

    let descriptor_set_layout = {
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ]
        .map(|x| x.build());
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::empty())
            .bindings(&bindings);
        dev.create_descriptor_set_layout(&info, None).unwrap()
    };
    let layout = {
        let layouts = [descriptor_set_layout];
        let info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
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

    shared.lock().unwrap().compute_descriptor_set_layout = descriptor_set_layout;

    ComputeStage {
        pipeline,
        layout,
        shared,
    }
}

unsafe impl crate::Stage for ComputeStage {
    unsafe fn record_commands(&self, dev: &ash::Device, args: &crate::StageArgs) {
        let shared = self.shared.lock().unwrap();

        // compute
        for sh in shared.data_sets.iter() {
            let data = &sh.per_image[args.index].compute;
            dev.cmd_bind_pipeline(args.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                args.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[data.descriptor_set],
                &[],
            );
            dev.cmd_dispatch_indirect(args.cmd, data.parameters.0, 0);
        }

        // sync
        let mut buffer_memory_barriers = Vec::with_capacity(shared.data_sets.len());
        for sh in shared.data_sets.iter() {
            let data = &sh.per_image[args.index].compute;
            buffer_memory_barriers.push(
                vk::BufferMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(data.data.0)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .build(),
            );
        }
        dev.cmd_pipeline_barrier(
            args.cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::DependencyFlags::empty(),
            &[],
            &buffer_memory_barriers,
            &[],
        );
    }

    unsafe fn rebuild_swapchain(
        &mut self,
        dev: &mut crate::Dev,
        swapchain: &crate::swapchain::SwapChain,
    ) {
    }
}

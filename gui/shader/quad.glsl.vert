#version 460

layout (std430, push_constant) uniform Viewport {
	vec2 viewport;
};

layout (location = 0) in vec2 in_half_extents;
layout (location = 1) in vec2 in_rotation;
layout (location = 2) in vec2 in_position;
layout (location = 3) in vec2 in_uv_scale;
layout (location = 4) in vec2 in_uv_offset;
layout (location = 5) in uint in_texture_index;

layout (location = 0) out vec2 out_uv;
layout (location = 1) out uint out_texture_index;

void main() {
	vec2 uv = ivec2(gl_VertexIndex, gl_VertexIndex >> 1) & 1;

	out_uv = uv * in_uv_scale + in_uv_offset;

	gl_Position.zw = vec2(1);
	gl_Position.xy = in_half_extents * (2 * uv - 1);
	gl_Position.xy *= mat2(in_rotation, -in_rotation.y, in_rotation.x);
	gl_Position.y *= viewport.x / viewport.y;
	gl_Position.xy += in_position;

	out_texture_index = in_texture_index;
}

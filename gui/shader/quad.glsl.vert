#version 460

layout (std430, push_constant) uniform Viewport {
	vec2 viewport;
};

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_size;
layout (location = 2) in float in_rotation;
layout (location = 3) in uint in_texture_index;
layout (location = 4) in vec2 in_uv_start;
layout (location = 5) in vec2 in_uv_end;
layout (location = 6) in vec4 in_color;

layout (location = 0) out vec2 out_uv;
layout (location = 1) out uint out_texture_index;
layout (location = 2) out vec4 out_color;

void main() {
	vec2 uv = ivec2(gl_VertexIndex, gl_VertexIndex >> 1) & 1;

	out_uv = ((1 - uv) * in_uv_start) + (uv * in_uv_end);

	vec2 r = vec2(cos(in_rotation), sin(in_rotation));

	vec2 pos = in_size * uv;
	pos = mat2(r, -r.y, r.x) * pos;
	pos += in_position;
	pos = (pos / viewport) * 2 - vec2(1);

	gl_Position = vec4(pos, 1, 1);

	out_texture_index = in_texture_index;
	out_color = in_color;
}

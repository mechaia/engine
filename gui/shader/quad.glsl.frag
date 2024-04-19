#version 460

#extension GL_EXT_nonuniform_qualifier : enable

layout (location = 0) in vec2 in_uv;
layout (location = 1) in flat uint in_texture_index;
layout (location = 2) in flat vec4 in_color;

layout (binding = 0) uniform sampler2D textures_rgba[];

layout (location = 0) out vec4 out_color;

void main() {
	out_color = texture(textures_rgba[nonuniformEXT(in_texture_index)], in_uv);
	out_color *= in_color;
}

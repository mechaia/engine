#version 460

layout (binding = 0) uniform Camera {
    mat4 inv_projection;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 3) in uint material_index;

layout (location = 4) in mat4 projection;

layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out flat uint out_material_index;

void main() {
	gl_Position = projection * vec4(position, 1);

	out_position = (inv_projection * gl_Position).xyz;
	out_normal = (projection * vec4(normal, 0)).xyz;
	out_uv = uv;
    out_material_index = material_index;
}

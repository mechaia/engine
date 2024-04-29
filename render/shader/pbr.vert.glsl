#version 460

layout (binding = 0) uniform Camera {
    mat4 world_to_view;
    mat4 view_to_projection;
};

layout (binding = 2) readonly buffer Transforms {
	mat4 transforms[];
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in uvec4 joints;
layout (location = 4) in vec4 weights;

layout (location = 5) in uint transforms_offset;
layout (location = 6) in uint material_index;

layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out flat uint out_material_index;

void main() {
	vec3 pos = vec3(0);
	vec3 norm = vec3(0);

	for (uint i = 0; i < 4; i++) {
		mat4 world_to_view = transforms[transforms_offset + joints[i]];

		pos += weights[i] * (world_to_view * vec4(position, 1)).xyz;
		norm += weights[i] * (world_to_view * vec4(normal, 0)).xyz;
	}

	gl_Position = view_to_projection * vec4(pos, 1);

	out_position = pos;
	out_normal = norm;

	out_uv = uv;
	out_material_index = material_index;
}

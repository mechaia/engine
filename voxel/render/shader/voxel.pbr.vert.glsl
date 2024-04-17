#version 460

#define UV_MODE_LOCAL (0)
#define UV_MODE_WORLD (1)

layout (constant_id = 0) const uint UV_MODE = 0;

layout (binding = 0) uniform Camera {
    mat4 view_to_projection;
};

layout (location = 0) in uvec2 vertex_data;

layout (location = 1) in mat4 transform;

layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out flat uint out_material_index;

void main() {
	vec3 pos = uvec3(
		(vertex_data.x >> 0) & 0xfff,
		(vertex_data.x >> 12) & 0xfff,
		(vertex_data.y >> 0) & 0xfff
	);

	/* TODO figure out an intuitive way of converting vertex data
	float normal_data = (vertex_data >> 36) & 0x1f;
	vec3 normal;
	if (normal_data < 2) {
		
	} else {
	}
	*/
	uint norm_data = (vertex_data.y >> 12) & 0x7;
	vec3 norm = vec3((norm_data & 3) == 0, (norm_data & 3) == 1, (norm_data & 3) == 2);
	norm = ((norm_data >> 2) == 0) ? norm : -norm;

	uint material_index = vertex_data.y >> 20;


	pos = (transform * vec4(pos, 1)).xyz;
	norm = (transform * vec4(norm, 0)).xyz;



	gl_Position = view_to_projection * vec4(pos, 1);

	out_position = pos;
	out_normal = (view_to_projection * vec4(norm, 0)).xyz;

	out_material_index = material_index;

	switch (UV_MODE) {
	case UV_MODE_LOCAL: out_uv = vec2(pos.x + pos.y, pos.z); break;
	case UV_MODE_WORLD: out_uv = vec2(out_position.x + out_position.y, out_position.z); break;
	}
}

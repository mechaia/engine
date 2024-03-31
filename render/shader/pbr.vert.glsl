#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in mat4 projection;

layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out vec3 out_albedo;
layout (location = 4) out float out_metallic;
layout (location = 5) out float out_roughness;
layout (location = 6) out float out_ambient_occlusion;

void main() {
	gl_Position = projection * vec4(position, 1);
	// we will normalize in the fragment shader
	out_normal = (projection * vec4(normal, 0)).xyz;
	out_uv = uv;
}

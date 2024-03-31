#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in mat4 model_to_project;

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec2 out_uv;

void main() {
	gl_Position = model_to_project * vec4(position, 1);
	out_normal = (model_to_project * vec4(normal, 0)).xyz;
	out_uv = uv;
}

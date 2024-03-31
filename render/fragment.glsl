#version 450

layout (push_constant, std430) uniform Viewport {
	vec2 inv_viewport;
	float viewport_y_over_x;
};

layout (binding = 0) uniform Camera {
	mat4 inv_projection;
};

layout (location = 0) in vec3 normal_unnormalized;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec4 color;

void main() {
	vec3 normal = normalize(normal_unnormalized);
	color.w = 1;
	color.xyz = vec3(0.5);
	color.xyz += vec3(0.5) * max(dot(normal, normalize(-vec3(1))), 0);

	// map XYZ to [0; 1]
	vec4 position = vec4(gl_FragCoord.xy * inv_viewport, gl_FragCoord.z, 1);
	// map XY to [-1; 1]
	position.xy = (position.xy - 0.5) * 2;
	position /= gl_FragCoord.w;
	//position.z = 1 / gl_FragCoord.z;
	// map XYZ to view
	position = inv_projection * position;

	//position *= 4;
	position = position - floor(position);

	color.xyz = position.xyz;
}

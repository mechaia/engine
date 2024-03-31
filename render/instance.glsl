#version 450

layout (binding = 0) uniform Camera {
	mat4 projection;
};

struct Instance {
	vec3 position;
	float scale;
	vec3 rotation;
	uint instance;
};

layout(std430, binding = 1) readonly buffer In {
	Instance instances[];
};

layout(std430, binding = 2) writeonly buffer Out {
	mat4 projections[];
};

mat3 quat_to_axes(vec4 r) {
	// copied from https://docs.rs/glam/0.27.0/src/glam/f32/sse2/mat4.rs.html#181-202
	vec3 r2 = r.xyz * 2;
	float xx = r.x * r2.x;
	float xy = r.x * r2.y;
	float xz = r.x * r2.z;
	float yy = r.y * r2.y;
	float yz = r.y * r2.z;
	float zz = r.z * r2.z;
	vec3 w = r.w * r2;

	return mat3(
		vec3(1 - (yy + zz), xy + w.z, xz - w.y),
		vec3(xy - w.z, 1 - (xx + zz), yz + w.x),
		vec3(xz + w.y, yz - w.x, 1 - (xx + yy)));
}

void main() {
	uint index = gl_GlobalInvocationID.x;
	Instance inst = instances[index];

	vec4 rotation = vec4(inst.rotation, sqrt(1 - dot(inst.rotation, inst.rotation)));

	// copied from https://docs.rs/glam/0.27.0/src/glam/f32/sse2/mat4.rs.html#215-223
	mat3 axes = quat_to_axes(rotation) * inst.scale;

	projections[index] = projection * mat4(
		vec4(axes[0], 0),
		vec4(axes[1], 0),
		vec4(axes[2], 0),
		vec4(inst.position, 1));
}

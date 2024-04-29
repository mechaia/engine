#version 450

//layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) uniform Camera {
	mat4 world_to_view;
	mat4 view_to_projection;
};

struct Transform {
	vec4 rotation;
	vec3 position;
	float scale;
};

layout(std430, binding = 1) readonly buffer In {
	Transform in_transforms[];
};

layout(std430, binding = 2) writeonly buffer Out {
	mat4 out_transforms[];
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
	Transform trf = in_transforms[index];

	//vec4 rotation = vec4(inst.rotation, sqrt(max(0, 1 - dot(inst.rotation, inst.rotation))));
	vec4 rotation = trf.rotation;

	// copied from https://docs.rs/glam/0.27.0/src/glam/f32/sse2/mat4.rs.html#215-223
	mat3 axes = quat_to_axes(rotation);

	out_transforms[index] = world_to_view * mat4(
		vec4(axes[0], 0),
		vec4(axes[1], 0),
		vec4(axes[2], 0),
		vec4(trf.position, 1));
}

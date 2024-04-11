// Ripped straight from https://learnopengl.com/PBR/Lighting with no shame.
#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#define MAX_DIRECTIONAL_LIGHTS (1)
#define MAX_MATERIALS (4096)

const float PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

layout (push_constant, std430) uniform Viewport {
	vec2 inv_viewport;
	float viewport_y_over_x;
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct Material {
    vec4 albedo;
    float roughness;
    float metallic;
    float ambient_occlusion;
    float _padding;
    uint albedo_texture_index;
    uint roughness_texture_index;
    uint metallic_texture_index;
    uint ambient_occlusion_texture_index;
};

layout (std430, binding = 1) readonly buffer DirectionalLights {
    DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
};

/*
// TODO cluster point and cone lights
layout (binding = 3) readonly buffer Lights {
	float ops[];
};
 */

layout (set = 1, binding = 0) uniform sampler2D textures_rgba[];

layout (std430, set = 2, binding = 0) uniform Materials {
    Material materials[MAX_MATERIALS];
};
/*
layout (std430, set = 2, binding = 0) readonly buffer Materials {
    Material materials[];
};
*/

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal_unnormalized;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in flat uint in_material_index;

layout (location = 0) out vec4 out_color;

vec3 normal;
vec3 view_normal;

// material properties after texture sampling
vec3 albedo;
float roughness;
float metallic;

vec3 reflectivity;

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float DistributionGGX(vec3 N, vec3 H) {
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * (denom * denom);
	
    return num / denom;
}

vec3 fresnelSchlick(float cosTheta) {
    //return reflectivity + (1.0 - reflectivity) * pow(2.0, (-5.55473 * cosTheta - 6.98316) * cosTheta);
    return reflectivity + (1.0 - reflectivity) * pow5(clamp(1.0 - cosTheta, 0.0, 1.0));
}

float GeometrySchlickGGX(float NdotV) {
    float r = roughness + 1.0;
    float k = (r*r) / 8.0;

    return 1.0 / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 ray) {
    float ggx2 = GeometrySchlickGGX(max(dot(normal, view_normal), 0.0));
    float ggx1 = GeometrySchlickGGX(max(dot(normal, ray), 0.0));
    return ggx1 * ggx2;
}

// radiance equation
//
// "inlined" formulas, simplified to eliminate potential divide-by-zero in specular denominator.
// Should be good for performance too :)
vec3 calc_radiance(vec3 ray, vec3 light_color) {
    vec3 halfway_normal = normalize(ray + view_normal);

    // cook-torrance brdf
    float normal_distribution = DistributionGGX(normal, halfway_normal);
    float geometry = GeometrySmith(ray);
    vec3 fresnel = fresnelSchlick(max(dot(halfway_normal, view_normal), 0.0));

    vec3 specular_ratio = fresnel;
    vec3 diffuse_ratio = (vec3(1) - specular_ratio) * (1 - metallic);

    // https://computergraphics.stackexchange.com/questions/3946/
    vec3 specular = (normal_distribution * geometry * fresnel) / 4.0;

    vec3 diffuse = diffuse_ratio * albedo / PI;

    return (diffuse + specular) * light_color * max(dot(normal, ray), 0.0);
}

void main() {
    normal = normalize(normal_unnormalized);
    view_normal = normalize(position);

    Material m = materials[nonuniformEXT(in_material_index)];

    vec4 albedo_a = m.albedo * texture(textures_rgba[nonuniformEXT(m.albedo_texture_index)], in_uv);
    albedo = albedo_a.rgb;
    roughness = m.roughness * texture(textures_rgba[nonuniformEXT(m.roughness_texture_index)], in_uv).r;
    metallic = m.metallic * texture(textures_rgba[nonuniformEXT(m.metallic_texture_index)], in_uv).r;
    float ambient_occlusion = m.ambient_occlusion * texture(textures_rgba[nonuniformEXT(m.ambient_occlusion_texture_index)], in_uv).r;

    reflectivity = mix(vec3(0.04), albedo, metallic);

    vec3 outgoing_light = vec3(0);
    for (uint i = 0; i < directional_lights.length(); i++) {
        DirectionalLight light = directional_lights[i];
        outgoing_light += calc_radiance(light.direction.xyz, light.color.rgb);
    }

    vec3 ambient = vec3(0.03) * albedo * ambient_occlusion;
    vec3 color = ambient + outgoing_light;
	
    // gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    out_color = vec4(color, albedo_a.w);
}

// Ripped straight from https://learnopengl.com/PBR/Lighting with no shame.
#version 460

#define MAX_DIRECTIONAL_LIGHTS (1)
const float PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

layout (push_constant, std430) uniform Viewport {
	vec2 inv_viewport;
	float viewport_y_over_x;
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

layout (binding = 1) uniform sampler2D albedo_textures[];
layout (binding = 2) uniform sampler2D roughness_textures[];
layout (binding = 3) uniform sampler2D metallic_textures[];
layout (binding = 4) uniform sampler2D ambient_occlusion_textures[];

struct Material {
    vec2 uv_offset;
    vec2 uv_scale;
    vec3 albedo;
    uint albedo_texture_index;
    float roughness;
    uint roughness_texture_index;
    float metallic;
    uint metallic_texture_index;
    float ambient_occlusion;
    uint ambient_occlusion_texture_index;
};

layout (std430, binding = 5) readonly buffer Materials {
    Material materials[];
};

layout (std430, binding = 6) readonly buffer DirectionalLights {
    DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
};

/*
// TODO cluster point and cone lights
layout (binding = 0) readonly buffer Lights {
	float ops[];
};
 */

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal_unnormalized;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in flat uint material_index;

layout (location = 0) out vec4 out_color;

vec3 normal;
vec3 view_normal;

// material properties after texture sampling
vec3 albedo;
float roughness;
float metallic;
float ambient_occlusion;

vec3 reflectivity;

void sample_material() {
}

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
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV) {
    float r = roughness + 1.0;
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith(vec3 ray) {
    float ggx2 = GeometrySchlickGGX(max(dot(normal, view_normal), 0.0));
    float ggx1 = GeometrySchlickGGX(max(dot(normal, ray), 0.0));
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta) {
    //return reflectivity + (1.0 - reflectivity) * pow(2.0, (-5.55473 * cosTheta - 6.98316) * cosTheta);

    return reflectivity + (1.0 - reflectivity) * pow5(clamp(1.0 - cosTheta, 0.0, 1.0));
}

// radiance equation
vec3 calc_radiance(vec3 ray, vec3 light_color) {
    vec3 halfway_normal = normalize(ray + view_normal);

    // cook-torrance brdf
    float normal_distribution = DistributionGGX(normal, halfway_normal);
    float geometry = GeometrySmith(ray);
    vec3 fresnel = fresnelSchlick(max(dot(halfway_normal, view_normal), 0.0));

    vec3 specular_ratio = fresnel;
    vec3 diffuse_ratio = (vec3(1) - specular_ratio) * (1 - metallic);

    vec3 numerator = normal_distribution * geometry * fresnel;
    float eps = 1.0 / 256;
    eps = 0;
    // FIXME this shit is fucking me up with all the damn artifacts
    //float denominator = 4.0 * max(dot(normal, view_normal), 0.1) * max(dot(normal, ray), eps) + 0.001;
    float denominator = 4.0 * max(dot(normal, view_normal), 0.0) * max(dot(normal, ray), eps) + 0.000;
    //float denominator = 4.0 * abs(dot(normal, view_normal) * dot(normal, ray)) + 0.01;
    /*
    if (dot(normal, ray) <= 0.0 && dot(normal, view_normal) <= 0.0)
        return vec3(100, 100, 0);
    if (dot(normal, ray) <= 0.0)
        return vec3(100, 0, 0);
    if (dot(normal, view_normal) <= 0.0)
        return vec3(0, 100, 0);
        */
    /*
    denominator = max(dot(normal, view_normal), 0.0);
    denominator = max(dot(normal, ray), 0.0);
    */
    vec3 specular = numerator / denominator;

    vec3 diffuse = diffuse_ratio * albedo / PI;

    return (diffuse + specular) * light_color * max(dot(normal, ray), 0.0);
}


float GeometrySchlickGGX2(float NdotV) {
    float r = roughness + 1.0;
    float k = (r*r) / 8.0;

    float num   = 1;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith2(vec3 ray) {
    float ggx2 = GeometrySchlickGGX2(max(dot(normal, view_normal), 0.0));
    float ggx1 = GeometrySchlickGGX2(max(dot(normal, ray), 0.0));
    return ggx1 * ggx2;
}

// radiance equation
//
// "inlined" formulas, simplified to eliminate potential divide-by-zero in specular denominator.
// Should be good for performance too :)
vec3 calc_radiance2(vec3 ray, vec3 light_color) {
    vec3 halfway_normal = normalize(ray + view_normal);

    // cook-torrance brdf
    float normal_distribution = DistributionGGX(normal, halfway_normal);
    float geometry = GeometrySmith(ray);
    vec3 fresnel = fresnelSchlick(max(dot(halfway_normal, view_normal), 0.0));

    vec3 specular_ratio = fresnel;
    vec3 diffuse_ratio = (vec3(1) - specular_ratio) * (1 - metallic);

    vec3 numerator = normal_distribution * geometry * fresnel;
    // https://computergraphics.stackexchange.com/questions/3946/
    float denominator = 4.0;
    vec3 specular = numerator / denominator;

    vec3 diffuse = diffuse_ratio * albedo / PI;

    return (diffuse + specular) * light_color * max(dot(normal, ray), 0.0);
}

void main() {
    normal = normalize(normal_unnormalized);
    view_normal = normalize(position);

    Material m = materials[material_index];
    vec2 uv = in_uv * m.uv_scale + m.uv_offset;

    // cancel texture index if non-uniform indexing not supported
#if 0
# define albedo_tex m.albedo_texture_index
# define roughness_tex m.roughness_texture_index
# define metallic_tex m.metallic_texture_index
# define ambient_occlusion_tex m.ambient_occlusion_texture_index
#else
# define albedo_tex 0
# define roughness_tex 0
# define metallic_tex 0
# define ambient_occlusion_tex 0
#endif

    albedo = m.albedo * texture(albedo_textures[albedo_tex], uv).rgb;
    roughness = m.roughness * texture(roughness_textures[roughness_tex], uv).r;
    metallic = m.metallic * texture(metallic_textures[metallic_tex], uv).r;
    ambient_occlusion = m.ambient_occlusion * texture(ambient_occlusion_textures[ambient_occlusion_tex], uv).r;

    reflectivity = mix(vec3(0.04), albedo, metallic);

#if 0
    out_color = vec4(position - floor(position), 1);
    out_color = vec4(normal, 1);
    out_color = vec4(abs(normal), 1);
    return;
#endif

    vec3 outgoing_light = vec3(0);
    for (uint i = 0; i < directional_lights.length(); i++) {
        DirectionalLight light = directional_lights[i];
        outgoing_light += calc_radiance2(light.direction.xyz, light.color.rgb);
    }

    vec3 ambient = vec3(0.03) * albedo * ambient_occlusion;
    vec3 color = ambient + outgoing_light;
	
    // gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    out_color = vec4(color, 1);
}

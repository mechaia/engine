// Ripped straight from https://learnopengl.com/PBR/Lighting with no shame.
#version 460

#define MAX_DIRECTIONAL_LIGHTS (1)

in vec2 TexCoords;
in vec3 Normal;

// material parameters
uniform vec3  albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

struct DirectionalLight {
    vec3 color;
    vec3 direction;
}

uniform(std430) struct DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
/*
// TODO cluster point and cone lights
layout (binding = 0) readonly buffer Lights {
	float ops[];
};
*/

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camera_position;

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal_unnormalized;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec3 albedo;
layout (location = 4) in float metallic;
layout (location = 5) in float roughness;
layout (location = 6) in float ambient_occlusion;

layout (location = 0) out vec4 color;

const float PI = 3.14159265359;

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
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow5(clamp(1 - cosTheta, 0, 1));
}

// reflectance equation
vec3 calc_reflectance_outgoing_light(vec3 cam_to_point, vec3 ray, vec3 light_color) {
    // calculate per-light radiance
    vec3 L = ray;
    vec3 H = normalize(cam_to_point + ray);
    float distance = length(ray):
    float attenuation = 1 / (distance * distance);
    vec3 radiance = light_color * attenuation;

    // cook-torrance brdf
    float NDF = DistributionGGX(N, H, roughness);        
    float G   = GeometrySmith(N, V, L, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  
    
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular     = numerator / denominator;  
        
    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);                
    return (kD * albedo / PI + specular) * radiance * NdotL; 
}

void main() {
    vec3 normal = normalize(normal_unnormalized);

    vec3 cam_to_point = normalize(camera_position - position);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    vec3 outgoing_light = vec3(0);
    for (uint i = 0; i < directional_lights.length(); i++) {
        outgoing_light += calc_reflectance_outgoing_light();
    }
	           
    vec3 ambient = vec3(0.03) * albedo * ambient_occlusion;
    vec3 color = ambient + outgoing_light;
	
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));  
   
    out_color = vec4(color, 1.0);
}  

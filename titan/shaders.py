"""GLSL 4.10 shader sources for the TITAN renderer."""

HDR = "#version 410 core\n"

COMMON = """
const float PI = 3.14159265359;
vec2 dir_uv(vec3 d){ return vec2(atan(d.z, d.x) / (2.0*PI) + 0.5, acos(clamp(d.y,-1.0,1.0)) / PI); }
float hash11(float p){ p = fract(p*0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash31(vec3 p){ p = fract(p*0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y)*p.z); }
float vnoise(vec3 p){
    vec3 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
    float n000=hash31(i), n100=hash31(i+vec3(1,0,0)), n010=hash31(i+vec3(0,1,0)), n110=hash31(i+vec3(1,1,0));
    float n001=hash31(i+vec3(0,0,1)), n101=hash31(i+vec3(1,0,1)), n011=hash31(i+vec3(0,1,1)), n111=hash31(i+vec3(1,1,1));
    return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y), mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);
}
float plane_lines(vec2 uv, float seed){
    uv *= 3.4;
    vec2 id = floor(uv), f = fract(uv);
    float h1 = hash31(vec3(id, seed)), h2 = hash31(vec3(id + 31.7, seed + 4.0));
    float w = max(0.010, fwidth(uv.x) * 0.9);
    float lx = step(0.45, h1) * smoothstep(w, 0.0, f.x) * step(f.y, 0.30 + 0.65 * h2);
    float ly = step(0.45, h2) * smoothstep(w, 0.0, f.y) * step(f.x, 0.30 + 0.65 * h1);
    return max(lx, ly);
}
float panel_lines(vec3 p, vec3 n){
    vec3 w = pow(abs(n), vec3(8.0)); w /= (w.x + w.y + w.z + 1e-4);
    return w.x * plane_lines(p.yz, 1.0) + w.y * plane_lines(p.xz, 2.0) + w.z * plane_lines(p.xy, 3.0);
}
float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
"""

# ---------------------------------------------------------------- fullscreen
FS_VERT = HDR + """
out vec2 v_uv;
void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p*2.0 - 1.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------- environment
STUDIO = """
float sb(float az, float el, float caz, float cel, float haz, float hel, float soft){
    float da = mod(az - caz + PI, 2.0*PI) - PI;
    return smoothstep(haz + soft, haz - soft, abs(da)) * smoothstep(hel + soft, hel - soft, abs(el - cel));
}
vec3 studio(vec3 d, bool boxes){
    float el = asin(clamp(d.y,-1.0,1.0));
    float az = atan(d.z, d.x);
    vec3 top = vec3(0.0035,0.0050,0.0090), hor = vec3(0.0030,0.0065,0.0105), bot = vec3(0.0015,0.0025,0.0040);
    vec3 col = mix(hor, top, smoothstep(0.0, 1.1, el));
    col = mix(col, bot, smoothstep(0.0, -0.45, el));
    col += vec3(0.00,0.10,0.14) * exp(-pow(el*7.0,2.0)) * 0.05;
    if (boxes) {
        col += sb(az,el,  1.15, 0.80, 0.55,0.38,0.25) * vec3(1.00,0.93,0.82) * 9.0;  // key softbox
        col += sb(az,el,  2.30, 0.25, 0.20,0.55,0.15) * vec3(0.55,0.75,1.00) * 3.2;   // cool fill
        col += sb(az,el, -1.05, 0.35, 0.10,0.55,0.08) * vec3(0.25,0.85,1.00) * 9.0;  // rim cyan
        col += sb(az,el, -2.15, 0.35, 0.10,0.55,0.08) * vec3(1.00,0.30,0.18) * 7.0;  // rim warm
        col += sb(az,el, -1.57, 0.65, 0.60,0.30,0.20) * vec3(1.00,0.95,0.90) * 5.0;   // back key
        col += sb(az,el,  0.00, 0.20, 0.08,0.60,0.06) * vec3(0.80,0.90,1.00) * 3.0;   // side strips
        col += sb(az,el,   PI,  0.20, 0.08,0.60,0.06) * vec3(0.80,0.90,1.00) * 3.0;
        col += smoothstep(1.15, 1.50, el) * vec3(0.90,0.95,1.00) * 1.8;               // overhead panel
    }
    return col;
}
"""

ENVGEN_FS = HDR + COMMON + STUDIO + """
in vec2 v_uv; out vec4 o;
void main(){
    float az = (v_uv.x - 0.5) * 2.0 * PI; float th = v_uv.y * PI;
    vec3 d = vec3(cos(az)*sin(th), cos(th), sin(az)*sin(th));
    o = vec4(studio(d, true), 1.0);
}
"""

BG_FS = HDR + COMMON + STUDIO + """
in vec2 v_uv; out vec4 o;
uniform mat4 u_inv_vp; uniform vec3 u_cam; uniform float u_time; uniform float u_glow;
void main(){
    vec4 w = u_inv_vp * vec4(v_uv*2.0-1.0, 1.0, 1.0);
    vec3 d = normalize(w.xyz / w.w - u_cam);
    vec3 col = studio(d, false);
    float el = d.y;
    // distant arena ring lights
    float az = atan(d.z, d.x);
    float ring = smoothstep(0.004, 0.0, abs(el - 0.035));
    col += vec3(0.10,0.40,0.55) * ring * 0.22 * u_glow;
    col += vec3(0.004,0.02,0.03) * smoothstep(0.25, 0.0, abs(el)) * u_glow;
    o = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- mecha
MECHA_VS = HDR + """
uniform mat4 u_vp;
uniform mat4 u_bones[64];
uniform vec2 u_bone_fx[64];
in vec3 in_pos; in vec3 in_nrm; in float in_mid; in float in_bone;
out vec3 v_wpos; out vec3 v_nrm; out vec3 v_lpos; flat out float v_mid; flat out vec2 v_fx;
invariant gl_Position;
void main(){
    mat4 M = u_bones[int(in_bone + 0.5)];
    v_fx = u_bone_fx[int(in_bone + 0.5)];
    vec4 wp = M * vec4(in_pos, 1.0);
    v_wpos = wp.xyz; v_nrm = mat3(M) * in_nrm; v_lpos = in_pos; v_mid = in_mid;
    gl_Position = u_vp * wp;
}
"""

DEPTH_VS = HDR + """
uniform mat4 u_vp;
uniform mat4 u_bones[64];
in vec3 in_pos; in float in_bone;
void main(){ gl_Position = u_vp * (u_bones[int(in_bone + 0.5)] * vec4(in_pos, 1.0)); }
"""
DEPTH_FS = HDR + "void main(){}\n"

GBUF_VS = HDR + """
uniform mat4 u_vp; uniform mat4 u_view;
uniform mat4 u_bones[64];
in vec3 in_pos; in vec3 in_nrm; in float in_bone;
out vec3 v_vn; out vec3 v_wpos;
invariant gl_Position;
void main(){
    mat4 M = u_bones[int(in_bone + 0.5)];
    vec4 wp = M * vec4(in_pos, 1.0);
    v_vn = mat3(u_view) * (mat3(M) * in_nrm);
    v_wpos = wp.xyz;
    gl_Position = u_vp * wp;
}
"""
GBUF_FS = HDR + """
in vec3 v_vn; in vec3 v_wpos; out vec4 o;
uniform float u_clip_y;
void main(){ if (v_wpos.y < u_clip_y) discard; o = vec4(normalize(v_vn)*0.5+0.5, 1.0); }
"""

PBR_LIB = """
uniform vec3 u_cam;
uniform sampler2D u_env; uniform float u_env_lod; uniform float u_irr_lod; uniform float u_env_int;
uniform sampler2DShadow u_shadow; uniform mat4 u_shadow_vp; uniform float u_shadow_texel;
uniform vec3 u_key_dir; uniform vec3 u_key_col;
uniform vec3 u_rim1_dir; uniform vec3 u_rim1_col;
uniform vec3 u_rim2_dir; uniform vec3 u_rim2_col;
uniform float u_time; uniform float u_diff_int;

const vec2 PD[12] = vec2[](vec2(-0.326,-0.406), vec2(-0.840,-0.074), vec2(-0.696,0.457), vec2(-0.203,0.621),
    vec2(0.962,-0.195), vec2(0.473,-0.480), vec2(0.519,0.767), vec2(0.185,-0.893),
    vec2(0.507,0.064), vec2(0.896,0.412), vec2(-0.322,-0.933), vec2(-0.792,-0.598));

float shadow_at(vec3 wp, vec3 N){
    vec3 p = wp + N * (u_shadow_texel * 3.0);
    vec4 sp = u_shadow_vp * vec4(p, 1.0);
    vec3 sc = sp.xyz / sp.w * 0.5 + 0.5;
    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return 1.0;
    float ang = ign(gl_FragCoord.xy) * 6.2831853;
    mat2 rot = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    float s = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 o = rot * PD[i] * u_shadow_texel * 2.2;
        s += texture(u_shadow, vec3(sc.xy + o, sc.z - 0.0006));
    }
    return s / 12.0;
}
float D_GGX(float NoH, float a){ float a2 = a*a; float d = NoH*NoH*(a2-1.0)+1.0; return a2/(PI*d*d + 1e-6); }
float V_Smith(float NoV, float NoL, float a){
    float k = a*0.5;
    float gv = NoV/(NoV*(1.0-k)+k), gl = NoL/(NoL*(1.0-k)+k);
    return gv*gl/(4.0*NoV*NoL + 1e-4);
}
vec3 F_Schlick(vec3 F0, float VoH){ return F0 + (1.0 - F0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0); }
vec3 env_brdf(vec3 F0, float rough, float NoV){
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x*r.x, exp2(-9.28*NoV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return F0 * AB.x + AB.y;
}
vec3 direct_light(vec3 N, vec3 V, vec3 L, vec3 rad, vec3 albedo, float metal, float rough, vec3 F0, float coat){
    float NoL = max(dot(N, L), 0.0);
    if (NoL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NoV = max(dot(N, V), 1e-3), NoH = max(dot(N, H), 0.0), VoH = max(dot(V, H), 0.0);
    float a = max(rough*rough, 0.003);
    vec3 F = F_Schlick(F0, VoH);
    vec3 spec = D_GGX(NoH, a) * V_Smith(NoV, NoL, a) * F;
    vec3 diff = (1.0 - F) * (1.0 - metal) * albedo / PI;
    vec3 c = (diff + spec) * rad * NoL;
    if (coat > 0.0) {
        float ac = 0.012;
        vec3 Fc = F_Schlick(vec3(0.04), VoH) * coat;
        c = c * (1.0 - Fc) + D_GGX(NoH, ac) * V_Smith(NoV, NoL, ac) * Fc * rad * NoL;
    }
    return c;
}
"""

MECHA_FS = HDR + COMMON + PBR_LIB + """
in vec3 v_wpos; in vec3 v_nrm; in vec3 v_lpos; flat in float v_mid; flat in vec2 v_fx;
out vec4 o;
uniform vec3 u_albedo[16]; uniform vec4 u_params[16]; uniform vec3 u_fx_col;
uniform sampler2D u_ao; uniform vec2 u_res; uniform float u_use_ao;
uniform float u_emis_boost; uniform float u_clip_y;
uniform float u_charge;     // 0..1 palm / reactor charge glow
uniform float u_wear;

void main(){
    if (v_wpos.y < u_clip_y) discard;
    // materialise / dissolve effect for reveal sequence
    float burn = 0.0;
    if (v_fx.y > 0.001) {
        float dn = vnoise(v_lpos * 5.5) * 0.65 + vnoise(v_lpos * 16.0) * 0.35;
        if (dn < v_fx.y) discard;
        burn = smoothstep(v_fx.y + 0.12, v_fx.y, dn);
    }
    int mid = int(v_mid + 0.5);
    vec3 albedo = u_albedo[mid];
    vec4 prm = u_params[mid];
    float metal = prm.x, rough = prm.y, emis = prm.z, coat = prm.w;
    vec3 N = normalize(v_nrm);
    vec3 Ng = N;
    vec3 V = normalize(u_cam - v_wpos);
    if (dot(N, V) < 0.0 && !gl_FrontFacing) N = -N;

    // procedural micro detail (bone-local so it sticks to the part)
    vec3 lp = v_lpos;
    float n1 = vnoise(lp * 38.0), n2 = vnoise(lp * 120.0), n3 = vnoise(lp * 9.0);
    if (metal > 0.5) {
        float brush = vnoise(vec3(lp.x*3.0, lp.y*70.0, lp.z*3.0));
        rough = clamp(rough + (n2 - 0.5) * 0.06 + (brush - 0.5) * 0.06, 0.04, 1.0);
        vec3 g = vec3(vnoise(lp*22.0 + vec3(1.7,0,0)) - 0.5, vnoise(lp*22.0 + vec3(0,3.1,0)) - 0.5, vnoise(lp*22.0 + vec3(0,0,5.3)) - 0.5);
        N = normalize(N + g * 0.018);
    } else if (coat > 0.0) {
        rough = clamp(rough + (n3 - 0.5) * 0.10, 0.05, 1.0);
        // painted armour: edge wear exposes bare metal at tight bevels
        float curv = length(fwidth(Ng)) / (length(fwidth(v_wpos)) + 1e-5);
        float chip = smoothstep(14.0, 30.0, curv) * smoothstep(0.45, 0.8, vnoise(lp*30.0)) * 0.55 * u_wear;
        albedo = mix(albedo, vec3(0.42,0.43,0.46), chip);
        metal = mix(metal, 1.0, chip); rough = mix(rough, 0.35, chip);
        // scribed panel lines + dust that gathers near the floor
        float pl = panel_lines(lp, Ng);
        albedo *= 1.0 - 0.55 * pl;
        rough = clamp(rough + 0.25 * pl, 0.0, 1.0);
        albedo *= 1.0 - 0.10 * smoothstep(0.55, 0.9, n3) * u_wear;
        albedo *= 1.0 - 0.40 * smoothstep(0.55, 0.0, v_wpos.y) * (0.5 + n1) * u_wear;
    }

    float NoV = max(dot(N, V), 1e-3);
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    vec3 diffC = albedo * (1.0 - metal);

    float ao = 1.0;
    if (u_use_ao > 0.5) ao = texture(u_ao, gl_FragCoord.xy / u_res).r;
    float so = clamp(pow(NoV + ao, exp2(-16.0*rough - 1.0)) - 1.0 + ao, 0.0, 1.0);

    // direct lights
    float sh = shadow_at(v_wpos, Ng);
    vec3 col = vec3(0.0);
    col += direct_light(N, V, normalize(u_key_dir), u_key_col, albedo, metal, rough, F0, coat) * sh;
    col += direct_light(N, V, normalize(u_rim1_dir), u_rim1_col, albedo, metal, rough, F0, coat) * mix(0.35, 1.0, sh);
    col += direct_light(N, V, normalize(u_rim2_dir), u_rim2_col, albedo, metal, rough, F0, coat) * mix(0.35, 1.0, sh);

    // image based lighting
    vec3 R = reflect(-V, N);
    vec3 irr = textureLod(u_env, dir_uv(N), u_irr_lod).rgb;
    vec3 envS = textureLod(u_env, dir_uv(R), rough * u_env_lod).rgb;
    vec3 kS = env_brdf(F0, rough, NoV);
    col += (diffC * irr * (1.0 - kS) * ao * u_diff_int + envS * kS * so) * u_env_int;
    if (coat > 0.0) {
        vec3 envC = textureLod(u_env, dir_uv(R), 0.08 * u_env_lod).rgb;
        col += envC * env_brdf(vec3(0.04), 0.08, NoV) * coat * so * u_env_int * 0.7;
    }

    // emissive
    if (emis > 0.0) col += albedo * emis * (u_emis_boost + u_charge * 1.5);
    // hologram rim + scan lines + burn edge while parts materialise / assemble
    float holo = v_fx.x;
    if (holo > 0.001 || burn > 0.0) {
        float rim = pow(1.0 - NoV, 2.2);
        float scan = 0.5 + 0.5 * sin(v_wpos.y * 38.0 - u_time * 6.0);
        col += u_fx_col * (holo * (0.12 + 1.5 * rim + 0.25 * scan) + burn * 4.0);
    }

    // light atmospheric haze toward the horizon colour
    float dist = length(u_cam - v_wpos);
    col = mix(col, vec3(0.0030,0.0065,0.0105), 1.0 - exp(-dist*0.004));
    o = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- floor
FLOOR_VS = HDR + """
uniform mat4 u_vp;
in vec2 in_xz;
out vec3 v_wpos;
invariant gl_Position;
void main(){ v_wpos = vec3(in_xz.x, 0.0, in_xz.y); gl_Position = u_vp * vec4(v_wpos, 1.0); }
"""
FLOOR_GBUF_FS = HDR + """
uniform mat4 u_view; out vec4 o;
void main(){ o = vec4(normalize(mat3(u_view) * vec3(0.0,1.0,0.0))*0.5+0.5, 1.0); }
"""
FLOOR_FS = HDR + COMMON + PBR_LIB + """
in vec3 v_wpos; out vec4 o;
uniform sampler2D u_refl; uniform vec2 u_res;
uniform sampler2D u_ao; uniform float u_use_ao;
uniform vec3 u_focus; uniform vec3 u_accent;
uniform vec3 u_target; uniform float u_target_on;
uniform float u_power;
float gridline(vec2 p, float w){
    vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p);
    return 1.0 - clamp(min(g.x, g.y) / w, 0.0, 1.0);
}
void main(){
    vec3 V = normalize(u_cam - v_wpos);
    float dist = length(v_wpos.xz);
    // surface: polished dark composite tile with slight roughness variation
    vec2 p = v_wpos.xz;
    float tile = vnoise(vec3(floor(p*0.5), 1.0));
    float micro = vnoise(vec3(p*9.0, 2.0));
    float rough = 0.22 + 0.04 * tile;
    vec3 albedo = vec3(0.030, 0.034, 0.042) * (0.85 + 0.3 * tile) * (0.96 + 0.08 * micro);
    vec3 N = vec3(0.0, 1.0, 0.0);
    float NoV = max(dot(N, V), 1e-3);
    float F = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);
    F = clamp(F * 1.6 + 0.04, 0.0, 1.0);

    // planar reflection (blurred, stretches with roughness)
    vec2 uv = gl_FragCoord.xy / u_res;
    vec2 off = vec2(0.0);
    float blur = rough * 0.010 + 0.0015;
    vec3 refl = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 o2 = PD[i] * blur;
        float w = 1.0 - 0.4 * length(PD[i]);
        refl += texture(u_refl, uv + off + o2 * vec2(u_res.y / u_res.x, 1.6)).rgb * w;
        wsum += w;
    }
    refl /= wsum;

    float sh = shadow_at(v_wpos, vec3(0,1,0));
    vec3 col = vec3(0.0);
    col += direct_light(N, V, normalize(u_key_dir), u_key_col, albedo, 0.0, rough, vec3(0.04), 0.0) * sh;
    float ao = 1.0;
    if (u_use_ao > 0.5) ao = texture(u_ao, uv).r;
    vec3 irr = textureLod(u_env, dir_uv(N), u_irr_lod).rgb;
    col += albedo * irr * ao * u_env_int * u_diff_int;
    col += refl * F * 0.8 * ao;

    // emissive layout: fine grid + bold 5m grid + arena rings
    float fade = exp(-dist*0.075);
    float g1 = gridline(p, 0.9) * 0.035;
    float g5 = gridline(p * 0.2, 1.2) * 0.12;
    float r = dist;
    float ring_a = smoothstep(0.07, 0.0, abs(r - 6.0)) * 0.55;
    float ring_b = smoothstep(0.05, 0.0, abs(r - 6.6)) * 0.25;
    float ring_c = smoothstep(0.09, 0.0, abs(r - 12.0)) * 0.30;
    float ticks = smoothstep(0.03, 0.0, abs(fract(atan(p.y, p.x) * 36.0 / (2.0*PI)) - 0.5) - 0.46) * smoothstep(0.35, 0.0, abs(r - 6.3)) * 0.0;
    vec3 acc = u_accent;
    col += acc * (g1 + g5) * fade * (1.0 + 0.4 * u_power);
    col += acc * (ring_a + ring_b + ring_c) * fade * 1.4 * (1.0 + 0.8 * u_power);

    // soft pool of light under the robot
    float pool = exp(-dot(v_wpos.xz - u_focus.xz, v_wpos.xz - u_focus.xz) * 0.09);
    col += acc * pool * 0.012 * (1.0 + 2.5 * u_power);

    // movement target marker
    if (u_target_on > 0.5) {
        float dt = length(v_wpos.xz - u_target.xz);
        float ma = smoothstep(0.05, 0.0, abs(dt - 0.62 - 0.05*sin(u_time*4.0)));
        float mb = smoothstep(0.03, 0.0, abs(dt - 0.38));
        float mc = smoothstep(0.30, 0.0, dt) * 0.18;
        float ang = atan(v_wpos.z - u_target.z, v_wpos.x - u_target.x);
        float brk = step(0.5, fract(ang * 4.0 / PI + u_time * 0.25));
        col += acc * (ma * brk * 1.8 + mb * 0.8 + mc) * 1.5;
    }

    float dfog = 1.0 - exp(-length(u_cam - v_wpos) * 0.012);
    col = mix(col, vec3(0.0030,0.0065,0.0105), dfog);
    o = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- SSAO + bloom + composite
SSAO_FS = HDR + COMMON + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_depth; uniform sampler2D u_normal;
uniform mat4 u_proj; uniform mat4 u_inv_proj; uniform float u_radius; uniform float u_strength;
const vec3 K[16] = vec3[](
 vec3( 0.04,  0.02, 0.03), vec3(-0.07,  0.03, 0.05), vec3( 0.03, -0.09, 0.06), vec3( 0.10,  0.07, 0.09),
 vec3(-0.12, -0.05, 0.10), vec3( 0.05,  0.15, 0.12), vec3(-0.09,  0.13, 0.17), vec3( 0.17, -0.10, 0.14),
 vec3( 0.22,  0.10, 0.21), vec3(-0.21, -0.17, 0.20), vec3( 0.06, -0.28, 0.26), vec3(-0.27,  0.18, 0.28),
 vec3( 0.33,  0.07, 0.33), vec3(-0.16,  0.36, 0.38), vec3( 0.38, -0.30, 0.42), vec3(-0.45, -0.20, 0.46));
vec3 vpos(vec2 uv, float d){
    vec4 ndc = vec4(uv*2.0-1.0, d*2.0-1.0, 1.0);
    vec4 v = u_inv_proj * ndc; return v.xyz / v.w;
}
void main(){
    float d = texture(u_depth, v_uv).r;
    if (d >= 0.99999) { o = vec4(1.0); return; }
    vec3 P = vpos(v_uv, d);
    vec3 N = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    float a = ign(gl_FragCoord.xy) * 6.2831853;
    vec3 rv = vec3(cos(a), sin(a), 0.0);
    vec3 T = normalize(rv - N * dot(rv, N));
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    float occ = 0.0;
    for (int i = 0; i < 16; i++) {
        vec3 sp = P + TBN * K[i] * u_radius * 2.2;
        vec4 pp = u_proj * vec4(sp, 1.0);
        vec2 suv = pp.xy / pp.w * 0.5 + 0.5;
        float sd = texture(u_depth, suv).r;
        if (sd >= 0.99999) continue;
        float sz = vpos(suv, sd).z;
        float range = smoothstep(0.0, 1.0, u_radius / (abs(P.z - sz) + 1e-4));
        occ += (sz >= sp.z + 0.02 ? 1.0 : 0.0) * range;
    }
    float ao = 1.0 - occ / 16.0 * u_strength;
    o = vec4(vec3(clamp(ao, 0.0, 1.0)), 1.0);
}
"""

AOBLUR_FS = HDR + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_ao; uniform sampler2D u_depth; uniform vec2 u_texel;
void main(){
    float cd = texture(u_depth, v_uv).r;
    float s = 0.0, w = 0.0;
    for (int x = -2; x <= 2; x++) for (int y = -2; y <= 2; y++) {
        vec2 uv = v_uv + vec2(x, y) * u_texel;
        float dd = texture(u_depth, uv).r;
        float wt = exp(-abs(dd - cd) * 4000.0);
        s += texture(u_ao, uv).r * wt; w += wt;
    }
    o = vec4(vec3(s / max(w, 1e-4)), 1.0);
}
"""

BLOOM_DOWN_FS = HDR + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_prefilter; uniform float u_threshold;
vec3 prefilter(vec3 c){
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - u_threshold + 0.5, 0.0, 1.0); soft = soft*soft*0.25*2.0;
    float contrib = max(soft, br - u_threshold) / max(br, 1e-4);
    return min(c * contrib, vec3(60.0));
}
void main(){
    vec2 t = u_texel;
    vec3 a = texture(u_src, v_uv + t*vec2(-2,-2)).rgb, b = texture(u_src, v_uv + t*vec2(0,-2)).rgb, c = texture(u_src, v_uv + t*vec2(2,-2)).rgb;
    vec3 d = texture(u_src, v_uv + t*vec2(-2,0)).rgb,  e = texture(u_src, v_uv).rgb,                 f = texture(u_src, v_uv + t*vec2(2,0)).rgb;
    vec3 g = texture(u_src, v_uv + t*vec2(-2,2)).rgb,  h = texture(u_src, v_uv + t*vec2(0,2)).rgb,  i = texture(u_src, v_uv + t*vec2(2,2)).rgb;
    vec3 j = texture(u_src, v_uv + t*vec2(-1,-1)).rgb, k = texture(u_src, v_uv + t*vec2(1,-1)).rgb;
    vec3 l = texture(u_src, v_uv + t*vec2(-1,1)).rgb,  m = texture(u_src, v_uv + t*vec2(1,1)).rgb;
    vec3 r = e*0.125 + (a+c+g+i)*0.03125 + (b+d+f+h)*0.0625 + (j+k+l+m)*0.125;
    if (u_prefilter > 0.5) r = prefilter(r);
    o = vec4(r, 1.0);
}
"""
BLOOM_UP_FS = HDR + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_weight;
void main(){
    vec2 t = u_texel;
    vec3 r = texture(u_src, v_uv + t*vec2(-1,-1)).rgb + texture(u_src, v_uv + t*vec2(0,-1)).rgb*2.0 + texture(u_src, v_uv + t*vec2(1,-1)).rgb
           + texture(u_src, v_uv + t*vec2(-1,0)).rgb*2.0 + texture(u_src, v_uv).rgb*4.0 + texture(u_src, v_uv + t*vec2(1,0)).rgb*2.0
           + texture(u_src, v_uv + t*vec2(-1,1)).rgb + texture(u_src, v_uv + t*vec2(0,1)).rgb*2.0 + texture(u_src, v_uv + t*vec2(1,1)).rgb;
    o = vec4(r / 16.0 * u_weight, 1.0);
}
"""

COMPOSITE_FS = HDR + COMMON + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_hdr; uniform sampler2D u_bloom;
uniform float u_bloom_amt; uniform float u_exposure; uniform float u_time;
uniform float u_aberr; uniform float u_flash; uniform vec3 u_flash_col; uniform float u_vignette;
vec3 aces(vec3 x){
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
void main(){
    vec2 c = v_uv - 0.5;
    float r2 = dot(c, c);
    vec2 sh = c * u_aberr * (0.002 + r2 * 0.016);
    vec3 hdr;
    hdr.r = texture(u_hdr, v_uv - sh).r; hdr.g = texture(u_hdr, v_uv).g; hdr.b = texture(u_hdr, v_uv + sh).b;
    vec3 bl = texture(u_bloom, v_uv).rgb;
    vec3 col = hdr + bl * u_bloom_amt;
    col = col * (1.0 + u_flash * 1.0) + u_flash_col * u_flash * 0.05;
    col *= u_exposure;
    col *= 1.0 - u_vignette * smoothstep(0.12, 0.62, r2) ;
    col = aces(col);
    col = pow(col, vec3(1.0/2.2));
    col += (hash31(vec3(gl_FragCoord.xy, floor(u_time*24.0))) - 0.5) * 0.012;
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    o = vec4(col, luma);
}
"""

FXAA_FS = HDR + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel;
void main(){
    vec3 rgbM = texture(u_src, v_uv).rgb;
    float lM = dot(rgbM, vec3(0.299,0.587,0.114));
    float lNW = dot(texture(u_src, v_uv + vec2(-1,-1)*u_texel).rgb, vec3(0.299,0.587,0.114));
    float lNE = dot(texture(u_src, v_uv + vec2( 1,-1)*u_texel).rgb, vec3(0.299,0.587,0.114));
    float lSW = dot(texture(u_src, v_uv + vec2(-1, 1)*u_texel).rgb, vec3(0.299,0.587,0.114));
    float lSE = dot(texture(u_src, v_uv + vec2( 1, 1)*u_texel).rgb, vec3(0.299,0.587,0.114));
    float lMin = min(lM, min(min(lNW,lNE), min(lSW,lSE)));
    float lMax = max(lM, max(max(lNW,lNE), max(lSW,lSE)));
    vec2 dir;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y =  ((lNW + lSW) - (lNE + lSE));
    float dirReduce = max((lNW+lNE+lSW+lSE) * 0.25 * 0.125, 1.0/128.0);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpMin, vec2(-8.0), vec2(8.0)) * u_texel;
    vec3 rgbA = 0.5 * (texture(u_src, v_uv + dir*(1.0/3.0-0.5)).rgb + texture(u_src, v_uv + dir*(2.0/3.0-0.5)).rgb);
    vec3 rgbB = rgbA*0.5 + 0.25*(texture(u_src, v_uv + dir*-0.5).rgb + texture(u_src, v_uv + dir*0.5).rgb);
    float lB = dot(rgbB, vec3(0.299,0.587,0.114));
    o = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
"""

# ---------------------------------------------------------------- HUD quad
HUD_VS = HDR + """
uniform vec4 u_rect;  // x, y, w, h in NDC (bottom-left origin)
uniform float u_flip;
in vec2 in_quad;
out vec2 v_uv;
void main(){
    v_uv = vec2(in_quad.x, mix(1.0 - in_quad.y, in_quad.y, u_flip));
    gl_Position = vec4(u_rect.xy + in_quad * u_rect.zw, 0.0, 1.0);
}
"""
HUD_FS = HDR + """
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform float u_alpha;
void main(){ vec4 c = texture(u_tex, v_uv); o = vec4(c.rgb, c.a * u_alpha); }
"""

# ---------------------------------------------------------------- FX
SPRITE_VS = HDR + """
uniform mat4 u_vp; uniform vec3 u_right; uniform vec3 u_up;
in vec2 in_quad;
in vec3 in_pos; in float in_size; in vec4 in_col; in float in_kind;
out vec2 v_q; out vec4 v_col; flat out float v_kind;
void main(){
    vec2 q = in_quad * 2.0 - 1.0;
    vec3 w = in_pos + (u_right * q.x + u_up * q.y) * in_size;
    v_q = q; v_col = in_col; v_kind = in_kind;
    gl_Position = u_vp * vec4(w, 1.0);
}
"""
SPRITE_FS = HDR + """
in vec2 v_q; in vec4 v_col; flat in float v_kind; out vec4 o;
uniform sampler2D u_depth; uniform vec2 u_res; uniform vec2 u_nf; uniform float u_soft;
float linz(float z){ return u_nf.x * u_nf.y / (u_nf.y - z * (u_nf.y - u_nf.x)); }
void main(){
    float r2 = dot(v_q, v_q);
    if (r2 > 1.0) discard;
    float k = v_kind;
    float fade = 1.0;
    if (u_soft > 0.5 && k > 0.5 && k < 1.5) {
        float sd = texture(u_depth, gl_FragCoord.xy / u_res).r;
        fade = clamp((linz(sd) - linz(gl_FragCoord.z)) / 0.7, 0.0, 1.0);
    }
    if (k < 0.5) {            // soft additive glow
        float p = exp(-r2 * 4.2) * (1.0 - r2);
        o = vec4(v_col.rgb * p * v_col.a, 0.0);
    } else if (k < 1.5) {     // smoke / dust (premultiplied alpha)
        float p = smoothstep(1.0, 0.0, r2);
        p *= p;
        float a = p * v_col.a * fade;
        o = vec4(v_col.rgb * a, a);
    } else {                  // hot spark core
        float p = exp(-r2 * 9.0);
        o = vec4(v_col.rgb * p * v_col.a, 0.0);
    }
}
"""

BEAM_VS = HDR + """
uniform mat4 u_vp; uniform vec3 u_p0; uniform vec3 u_p1; uniform float u_w0; uniform float u_w1; uniform vec3 u_cam;
in vec2 in_quad;
out vec2 v_q; out float v_t;
void main(){
    float t = in_quad.x;
    vec3 p = mix(u_p0, u_p1, t);
    vec3 dir = normalize(u_p1 - u_p0 + 1e-6);
    vec3 toc = normalize(u_cam - p);
    vec3 side = cross(dir, toc);
    float sl = length(side);
    side = sl > 1e-4 ? side / sl : vec3(0.0, 1.0, 0.0);
    float w = mix(u_w0, u_w1, t);
    p += side * w * (in_quad.y * 2.0 - 1.0);
    v_q = vec2(t, in_quad.y * 2.0 - 1.0); v_t = t;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""
BEAM_FS = HDR + COMMON + """
in vec2 v_q; in float v_t; out vec4 o;
uniform vec3 u_col; uniform vec3 u_core; uniform float u_int; uniform float u_time; uniform float u_mode; uniform float u_len;
void main(){
    float y = abs(v_q.y);
    float t = v_q.x;
    vec3 col;
    if (u_mode < 0.5) {  // laser
        float core = exp(-pow(y * 2.4, 2.0));
        float glow = exp(-y * 3.2);
        float flick = 0.85 + 0.15 * sin(u_time * 90.0 + t * u_len * 3.0) + 0.1 * vnoise(vec3(t * u_len * 2.0 - u_time * 30.0, y * 2.0, 1.0));
        col = (u_col * glow * 0.9 + u_core * core * 1.8) * flick;
        col *= smoothstep(0.0, 0.03, t) * smoothstep(1.0, 0.985, t) ;
    } else {             // flame / thrust plume
        float turb = vnoise(vec3(t * 7.0 - u_time * 22.0, v_q.y * 2.2, u_time * 4.0));
        float wob = (turb - 0.5) * 0.9 * t;
        float yy = abs(v_q.y + wob);
        float body = smoothstep(1.0, 0.0, yy) * smoothstep(1.0, 0.1, t);
        float core = smoothstep(0.55, 0.0, yy) * smoothstep(0.8, 0.0, t);
        col = u_col * body * (0.6 + turb) + u_core * core * 2.2;
        col *= smoothstep(0.0, 0.05, t);
    }
    o = vec4(col * u_int, 0.0);
}
"""

RING_VS = HDR + """
uniform mat4 u_vp; uniform vec3 u_center; uniform float u_extent;
in vec2 in_quad;
out vec3 v_wpos;
void main(){
    vec2 q = in_quad * 2.0 - 1.0;
    v_wpos = u_center + vec3(q.x, 0.0, q.y) * u_extent + vec3(0.0, 0.012, 0.0);
    gl_Position = u_vp * vec4(v_wpos, 1.0);
}
"""
RING_FS = HDR + COMMON + """
in vec3 v_wpos; out vec4 o;
uniform vec3 u_center; uniform float u_radius; uniform float u_thick; uniform vec3 u_col; uniform float u_int; uniform float u_fill;
uniform float u_time;
void main(){
    vec2 d = v_wpos.xz - u_center.xz;
    float r = length(d);
    float x = (r - u_radius) / max(u_thick, 1e-3);
    float ringp = exp(-x * x * 2.0) ;
    float inner = smoothstep(u_radius, 0.0, r) * u_fill;
    float ang = atan(d.y, d.x);
    float n = 0.75 + 0.25 * vnoise(vec3(ang * 3.0, r * 1.5, u_time * 5.0));
    float a = (ringp * n + inner) * u_int;
    o = vec4(u_col * a, 0.0);
}
"""

SHIELD_VS = HDR + """
uniform mat4 u_vp; uniform mat4 u_model;
in vec3 in_pos; in vec3 in_nrm;
out vec3 v_wpos; out vec3 v_nrm; out vec3 v_dir;
void main(){
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_wpos = w.xyz; v_nrm = normalize(mat3(u_model) * in_nrm); v_dir = normalize(in_pos);
    gl_Position = u_vp * w;
}
"""
SHIELD_FS = HDR + COMMON + """
in vec3 v_wpos; in vec3 v_nrm; in vec3 v_dir; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_col; uniform float u_int; uniform float u_time; uniform vec4 u_hit[4];
float hexd(vec2 p){
    p = abs(p); const vec2 s = vec2(1.0, 1.7320508);
    return max(dot(p, s*0.5), p.x);
}
float hexgrid(vec2 uv){
    const vec2 s = vec2(1.0, 1.7320508);
    vec4 hc = floor(vec4(uv, uv - vec2(1.0, 0.5)) / s.xyxy) + 0.5;
    vec4 h = vec4(uv - hc.xy * s, uv - (hc.zw + 0.5) * s);
    vec2 p = dot(h.xy, h.xy) < dot(h.zw, h.zw) ? h.xy : h.zw;
    return 0.5 - hexd(p);
}
void main(){
    vec3 V = normalize(u_cam - v_wpos);
    float fres = pow(1.0 - abs(dot(normalize(v_nrm), V)), 2.2);
    vec2 uv = vec2(atan(v_dir.z, v_dir.x) * 5.0, v_dir.y * 9.0);
    float hx = hexgrid(uv * 1.4);
    float edge = smoothstep(0.06, 0.0, hx);
    float cell = hash31(vec3(floor(uv * 1.4 * 2.0), 3.0));
    float pulse = 0.5 + 0.5 * sin(u_time * 2.0 + cell * 6.28);
    float scan = smoothstep(0.04, 0.0, abs(fract(v_dir.y * 1.6 - u_time * 0.35) - 0.5) - 0.46);
    float hit = 0.0;
    for (int i = 0; i < 4; i++) {
        if (u_hit[i].w < 1.2) {
            float ang = acos(clamp(dot(v_dir, normalize(u_hit[i].xyz)), -1.0, 1.0));
            float rr = u_hit[i].w * 1.8;
            hit += exp(-pow((ang - rr) * 6.0, 2.0)) * (1.0 - u_hit[i].w / 1.2);
        }
    }
    float a = fres * 1.0 + edge * (0.22 + 0.30 * pulse) * (0.3 + fres) + scan * 0.2 + hit * 1.4 + 0.012;
    o = vec4(u_col * a * u_int, 0.0);
}
"""

SOLID_VS = HDR + """
uniform mat4 u_vp; uniform mat4 u_model;
in vec3 in_pos; in vec3 in_nrm;
out vec3 v_n; out vec3 v_w;
void main(){ vec4 w = u_model * vec4(in_pos, 1.0); v_w = w.xyz; v_n = normalize(mat3(u_model) * in_nrm); gl_Position = u_vp * w; }
"""
SOLID_FS = HDR + """
in vec3 v_n; in vec3 v_w; out vec4 o;
uniform vec3 u_col; uniform float u_int; uniform vec3 u_cam;
void main(){
    vec3 V = normalize(u_cam - v_w);
    float f = 0.35 + 0.65 * pow(1.0 - abs(dot(normalize(v_n), V)), 2.0);
    o = vec4(u_col * u_int * f, 0.0);
}
"""

/**
 * WebGL Shaders for Simplex Noise and 3-Layer Neural Network GPU Math
 * Standard Stefan Gustavson / Ian McEwan 2D Simplex Noise (Ashima Arts)
 * Network input: Exactly TWO floating-point numbers (x, y)
 * Layer 1: 2 inputs -> 16 hidden neurons (4 vec4s)
 * Layer 2: 16 hidden neurons -> 16 hidden neurons
 * Layer 3: 16 hidden neurons -> 1 scalar output
 */

export const VERTEX_SHADER = `
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vPos;
uniform float uFrequency;
uniform vec2 uOffset;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  // Map [-1, 1] screen coordinates to centered coordinates (x, y) scaled by frequency
  vPos = (vUv - 0.5) * uFrequency + uOffset;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

export const SIMPLEX_NOISE_LIB = `
// 2D Simplex Noise by Stefan Gustavson / Ian McEwan
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
  const vec4 C = vec4(
    0.211324865405187,   // (3.0 - sqrt(3.0)) / 6.0
    0.366025403784439,   // 0.5 * (sqrt(3.0) - 1.0)
    -0.577350269189626,  // -1.0 + 2.0 * C.x
    0.024390243902439    // 1.0 / 41.0
  );

  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);

  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
        + i.x + vec3(0.0, i1.x, 1.0));

  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;

  return 130.0 * dot(m, g);
}
`;

export const COLORMAP_LIB = `
// Turbo Colormap
vec3 colormapTurbo(float x) {
  x = clamp(x, 0.0, 1.0);
  const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
  const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
  const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;

  return vec3(
    dot(v4, kRedVec4)   + dot(v2, kRedVec2),
    dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
    dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
  );
}

// Viridis Colormap
vec3 colormapViridis(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
  vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
  vec3 c2 = vec3(0.993248, 0.906157, 0.143936);
  if (t < 0.5) {
    return mix(c0, c1, t * 2.0);
  } else {
    return mix(c1, c2, (t - 0.5) * 2.0);
  }
}

// Cool-Warm Divergent Colormap (Blue to White to Red)
vec3 colormapCoolWarm(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 cool = vec3(0.18, 0.44, 0.85);
  vec3 mid  = vec3(0.95, 0.95, 0.95);
  vec3 warm = vec3(0.85, 0.18, 0.18);
  if (t < 0.5) {
    return mix(cool, mid, t * 2.0);
  } else {
    return mix(mid, warm, (t - 0.5) * 2.0);
  }
}

// Plasma Colormap
vec3 colormapPlasma(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.05, 0.03, 0.53);
  vec3 c1 = vec3(0.80, 0.22, 0.56);
  vec3 c2 = vec3(0.94, 0.97, 0.13);
  if (t < 0.5) {
    return mix(c0, c1, t * 2.0);
  } else {
    return mix(c1, c2, (t - 0.5) * 2.0);
  }
}

// Colormap selector
vec3 applyColormap(float val, int scheme) {
  val = clamp(val, 0.0, 1.0);
  if (scheme == 0) return vec3(val); // Grayscale
  if (scheme == 1) return colormapTurbo(val);
  if (scheme == 2) return colormapViridis(val);
  if (scheme == 3) return colormapCoolWarm(val);
  if (scheme == 4) return colormapPlasma(val);
  return vec3(val);
}
`;

export const ACTIVATION_LIB = `
float fastTanh(float x) {
  float exp2x = exp(clamp(2.0 * x, -12.0, 12.0));
  return (exp2x - 1.0) / (exp2x + 1.0);
}

float activate(float x, int type, float omega) {
  if (type == 0) { // Tanh
    return fastTanh(x);
  } else if (type == 1) { // Sinusoid (SIREN)
    return sin(omega * x);
  } else { // Sigmoid
    return 1.0 / (1.0 + exp(-clamp(x, -12.0, 12.0)));
  }
}
`;

/**
 * Fragment Shader for Canvas 1: Ground Truth Simplex Noise
 */
export const FRAGMENT_SHADER_SIMPLEX = `
precision highp float;
varying vec2 vUv;
varying vec2 vPos;

uniform int uColormap;
uniform float uContours;

${SIMPLEX_NOISE_LIB}
${COLORMAP_LIB}

void main() {
  // Input: 2 floating point numbers (vPos.x, vPos.y)
  float noise = snoise(vPos); // [-1.0, 1.0]
  float norm = clamp(noise * 0.5 + 0.5, 0.0, 1.0); // [0.0, 1.0]

  vec3 col = applyColormap(norm, uColormap);

  if (uContours > 0.5) {
    float c = abs(fract(norm * 10.0 - 0.5) - 0.5);
    float contour = smoothstep(0.04, 0.0, c);
    col = mix(col, vec3(0.0), contour * 0.35);
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

/**
 * Fragment Shader for Canvas 2: Three-Layer Neural Network Output
 * Network Input: EXACTLY TWO floating point numbers: (x.x, x.y)
 * Layer 1: 2 inputs -> 16 hidden neurons
 * Layer 2: 16 hidden neurons -> 16 hidden neurons
 * Layer 3: 16 hidden neurons -> 1 output neuron (prediction)
 */
export const FRAGMENT_SHADER_NEURAL_NET = `
precision highp float;
varying vec2 vUv;
varying vec2 vPos;

// Layer 1: 16 weights (vec2: x, y) + 16 biases
uniform vec2 uW1[16];
uniform float ub1[16];

// Layer 2: 16 neurons x 4 vec4s (64 vec4s = 256 weights) + 16 biases
uniform vec4 uW2_0[16];
uniform vec4 uW2_1[16];
uniform vec4 uW2_2[16];
uniform vec4 uW2_3[16];
uniform float ub2[16];

// Layer 3: 1 neuron x 4 vec4s (16 weights) + 1 bias
uniform vec4 uW3_0;
uniform vec4 uW3_1;
uniform vec4 uW3_2;
uniform vec4 uW3_3;
uniform float ub3;

uniform int uActivation;   // 0=Tanh, 1=Sine (SIREN), 2=Sigmoid
uniform float uOmega;      // Frequency multiplier for SIREN
uniform int uColormap;
uniform float uContours;

${ACTIVATION_LIB}
${COLORMAP_LIB}

void main() {
  // CONFIRMATION: Input to the neural network is strictly TWO floating point numbers: x.x and x.y
  vec2 x = vPos;

  // Layer 1: 2 floats -> 16 hidden units, grouped into 4 vec4s for SIMD hardware execution
  vec4 v1_0 = vec4(
    activate(dot(uW1[0], x) + ub1[0], uActivation, uOmega),
    activate(dot(uW1[1], x) + ub1[1], uActivation, uOmega),
    activate(dot(uW1[2], x) + ub1[2], uActivation, uOmega),
    activate(dot(uW1[3], x) + ub1[3], uActivation, uOmega)
  );
  vec4 v1_1 = vec4(
    activate(dot(uW1[4], x) + ub1[4], uActivation, uOmega),
    activate(dot(uW1[5], x) + ub1[5], uActivation, uOmega),
    activate(dot(uW1[6], x) + ub1[6], uActivation, uOmega),
    activate(dot(uW1[7], x) + ub1[7], uActivation, uOmega)
  );
  vec4 v1_2 = vec4(
    activate(dot(uW1[8], x) + ub1[8], uActivation, uOmega),
    activate(dot(uW1[9], x) + ub1[9], uActivation, uOmega),
    activate(dot(uW1[10], x) + ub1[10], uActivation, uOmega),
    activate(dot(uW1[11], x) + ub1[11], uActivation, uOmega)
  );
  vec4 v1_3 = vec4(
    activate(dot(uW1[12], x) + ub1[12], uActivation, uOmega),
    activate(dot(uW1[13], x) + ub1[13], uActivation, uOmega),
    activate(dot(uW1[14], x) + ub1[14], uActivation, uOmega),
    activate(dot(uW1[15], x) + ub1[15], uActivation, uOmega)
  );

  // Layer 2: 16 units -> 16 units (4 vec4s)
  vec4 v2_0 = vec4(
    activate(dot(uW2_0[0], v1_0) + dot(uW2_1[0], v1_1) + dot(uW2_2[0], v1_2) + dot(uW2_3[0], v1_3) + ub2[0], uActivation, uOmega),
    activate(dot(uW2_0[1], v1_0) + dot(uW2_1[1], v1_1) + dot(uW2_2[1], v1_2) + dot(uW2_3[1], v1_3) + ub2[1], uActivation, uOmega),
    activate(dot(uW2_0[2], v1_0) + dot(uW2_1[2], v1_1) + dot(uW2_2[2], v1_2) + dot(uW2_3[2], v1_3) + ub2[2], uActivation, uOmega),
    activate(dot(uW2_0[3], v1_0) + dot(uW2_1[3], v1_1) + dot(uW2_2[3], v1_2) + dot(uW2_3[3], v1_3) + ub2[3], uActivation, uOmega)
  );
  vec4 v2_1 = vec4(
    activate(dot(uW2_0[4], v1_0) + dot(uW2_1[4], v1_1) + dot(uW2_2[4], v1_2) + dot(uW2_3[4], v1_3) + ub2[4], uActivation, uOmega),
    activate(dot(uW2_0[5], v1_0) + dot(uW2_1[5], v1_1) + dot(uW2_2[5], v1_2) + dot(uW2_3[5], v1_3) + ub2[5], uActivation, uOmega),
    activate(dot(uW2_0[6], v1_0) + dot(uW2_1[6], v1_1) + dot(uW2_2[6], v1_2) + dot(uW2_3[6], v1_3) + ub2[6], uActivation, uOmega),
    activate(dot(uW2_0[7], v1_0) + dot(uW2_1[7], v1_1) + dot(uW2_2[7], v1_2) + dot(uW2_3[7], v1_3) + ub2[7], uActivation, uOmega)
  );
  vec4 v2_2 = vec4(
    activate(dot(uW2_0[8], v1_0) + dot(uW2_1[8], v1_1) + dot(uW2_2[8], v1_2) + dot(uW2_3[8], v1_3) + ub2[8], uActivation, uOmega),
    activate(dot(uW2_0[9], v1_0) + dot(uW2_1[9], v1_1) + dot(uW2_2[9], v1_2) + dot(uW2_3[9], v1_3) + ub2[9], uActivation, uOmega),
    activate(dot(uW2_0[10], v1_0) + dot(uW2_1[10], v1_1) + dot(uW2_2[10], v1_2) + dot(uW2_3[10], v1_3) + ub2[10], uActivation, uOmega),
    activate(dot(uW2_0[11], v1_0) + dot(uW2_1[11], v1_1) + dot(uW2_2[11], v1_2) + dot(uW2_3[11], v1_3) + ub2[11], uActivation, uOmega)
  );
  vec4 v2_3 = vec4(
    activate(dot(uW2_0[12], v1_0) + dot(uW2_1[12], v1_1) + dot(uW2_2[12], v1_2) + dot(uW2_3[12], v1_3) + ub2[12], uActivation, uOmega),
    activate(dot(uW2_0[13], v1_0) + dot(uW2_1[13], v1_1) + dot(uW2_2[13], v1_2) + dot(uW2_3[13], v1_3) + ub2[13], uActivation, uOmega),
    activate(dot(uW2_0[14], v1_0) + dot(uW2_1[14], v1_1) + dot(uW2_2[14], v1_2) + dot(uW2_3[14], v1_3) + ub2[14], uActivation, uOmega),
    activate(dot(uW2_0[15], v1_0) + dot(uW2_1[15], v1_1) + dot(uW2_2[15], v1_2) + dot(uW2_3[15], v1_3) + ub2[15], uActivation, uOmega)
  );

  // Layer 3: 16 units -> 1 scalar output prediction (outVal)
  float outVal = dot(uW3_0, v2_0) + dot(uW3_1, v2_1) + dot(uW3_2, v2_2) + dot(uW3_3, v2_3) + ub3;

  // Map network prediction to [0, 1] for colormapping
  float norm = clamp(outVal * 0.5 + 0.5, 0.0, 1.0);
  vec3 col = applyColormap(norm, uColormap);

  if (uContours > 0.5) {
    float c = abs(fract(norm * 10.0 - 0.5) - 0.5);
    float contour = smoothstep(0.04, 0.0, c);
    col = mix(col, vec3(0.0), contour * 0.35);
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

/**
 * Fragment Shader for Canvas 3: Difference / Error Heatmap
 * Computes | Simplex - NeuralNet | in real-time on GPU
 */
export const FRAGMENT_SHADER_DIFFERENCE = `
precision highp float;
varying vec2 vUv;
varying vec2 vPos;

// Layer 1
uniform vec2 uW1[16];
uniform float ub1[16];

// Layer 2
uniform vec4 uW2_0[16];
uniform vec4 uW2_1[16];
uniform vec4 uW2_2[16];
uniform vec4 uW2_3[16];
uniform float ub2[16];

// Layer 3
uniform vec4 uW3_0;
uniform vec4 uW3_1;
uniform vec4 uW3_2;
uniform vec4 uW3_3;
uniform float ub3;

uniform int uActivation;
uniform float uOmega;

${SIMPLEX_NOISE_LIB}
${ACTIVATION_LIB}

void main() {
  // Ground truth: 2 inputs (vPos.x, vPos.y)
  float groundTruth = snoise(vPos);

  // Neural network forward pass with 2 inputs (x.x, x.y)
  vec2 x = vPos;

  vec4 v1_0 = vec4(
    activate(dot(uW1[0], x) + ub1[0], uActivation, uOmega),
    activate(dot(uW1[1], x) + ub1[1], uActivation, uOmega),
    activate(dot(uW1[2], x) + ub1[2], uActivation, uOmega),
    activate(dot(uW1[3], x) + ub1[3], uActivation, uOmega)
  );
  vec4 v1_1 = vec4(
    activate(dot(uW1[4], x) + ub1[4], uActivation, uOmega),
    activate(dot(uW1[5], x) + ub1[5], uActivation, uOmega),
    activate(dot(uW1[6], x) + ub1[6], uActivation, uOmega),
    activate(dot(uW1[7], x) + ub1[7], uActivation, uOmega)
  );
  vec4 v1_2 = vec4(
    activate(dot(uW1[8], x) + ub1[8], uActivation, uOmega),
    activate(dot(uW1[9], x) + ub1[9], uActivation, uOmega),
    activate(dot(uW1[10], x) + ub1[10], uActivation, uOmega),
    activate(dot(uW1[11], x) + ub1[11], uActivation, uOmega)
  );
  vec4 v1_3 = vec4(
    activate(dot(uW1[12], x) + ub1[12], uActivation, uOmega),
    activate(dot(uW1[13], x) + ub1[13], uActivation, uOmega),
    activate(dot(uW1[14], x) + ub1[14], uActivation, uOmega),
    activate(dot(uW1[15], x) + ub1[15], uActivation, uOmega)
  );

  vec4 v2_0 = vec4(
    activate(dot(uW2_0[0], v1_0) + dot(uW2_1[0], v1_1) + dot(uW2_2[0], v1_2) + dot(uW2_3[0], v1_3) + ub2[0], uActivation, uOmega),
    activate(dot(uW2_0[1], v1_0) + dot(uW2_1[1], v1_1) + dot(uW2_2[1], v1_2) + dot(uW2_3[1], v1_3) + ub2[1], uActivation, uOmega),
    activate(dot(uW2_0[2], v1_0) + dot(uW2_1[2], v1_1) + dot(uW2_2[2], v1_2) + dot(uW2_3[2], v1_3) + ub2[2], uActivation, uOmega),
    activate(dot(uW2_0[3], v1_0) + dot(uW2_1[3], v1_1) + dot(uW2_2[3], v1_2) + dot(uW2_3[3], v1_3) + ub2[3], uActivation, uOmega)
  );
  vec4 v2_1 = vec4(
    activate(dot(uW2_0[4], v1_0) + dot(uW2_1[4], v1_1) + dot(uW2_2[4], v1_2) + dot(uW2_3[4], v1_3) + ub2[4], uActivation, uOmega),
    activate(dot(uW2_0[5], v1_0) + dot(uW2_1[5], v1_1) + dot(uW2_2[5], v1_2) + dot(uW2_3[5], v1_3) + ub2[5], uActivation, uOmega),
    activate(dot(uW2_0[6], v1_0) + dot(uW2_1[6], v1_1) + dot(uW2_2[6], v1_2) + dot(uW2_3[6], v1_3) + ub2[6], uActivation, uOmega),
    activate(dot(uW2_0[7], v1_0) + dot(uW2_1[7], v1_1) + dot(uW2_2[7], v1_2) + dot(uW2_3[7], v1_3) + ub2[7], uActivation, uOmega)
  );
  vec4 v2_2 = vec4(
    activate(dot(uW2_0[8], v1_0) + dot(uW2_1[8], v1_1) + dot(uW2_2[8], v1_2) + dot(uW2_3[8], v1_3) + ub2[8], uActivation, uOmega),
    activate(dot(uW2_0[9], v1_0) + dot(uW2_1[9], v1_1) + dot(uW2_2[9], v1_2) + dot(uW2_3[9], v1_3) + ub2[9], uActivation, uOmega),
    activate(dot(uW2_0[10], v1_0) + dot(uW2_1[10], v1_1) + dot(uW2_2[10], v1_2) + dot(uW2_3[10], v1_3) + ub2[10], uActivation, uOmega),
    activate(dot(uW2_0[11], v1_0) + dot(uW2_1[11], v1_1) + dot(uW2_2[11], v1_2) + dot(uW2_3[11], v1_3) + ub2[11], uActivation, uOmega)
  );
  vec4 v2_3 = vec4(
    activate(dot(uW2_0[12], v1_0) + dot(uW2_1[12], v1_1) + dot(uW2_2[12], v1_2) + dot(uW2_3[12], v1_3) + ub2[12], uActivation, uOmega),
    activate(dot(uW2_0[13], v1_0) + dot(uW2_1[13], v1_1) + dot(uW2_2[13], v1_2) + dot(uW2_3[13], v1_3) + ub2[13], uActivation, uOmega),
    activate(dot(uW2_0[14], v1_0) + dot(uW2_1[14], v1_1) + dot(uW2_2[14], v1_2) + dot(uW2_3[14], v1_3) + ub2[14], uActivation, uOmega),
    activate(dot(uW2_0[15], v1_0) + dot(uW2_1[15], v1_1) + dot(uW2_2[15], v1_2) + dot(uW2_3[15], v1_3) + ub2[15], uActivation, uOmega)
  );

  float netOut = dot(uW3_0, v2_0) + dot(uW3_1, v2_1) + dot(uW3_2, v2_2) + dot(uW3_3, v2_3) + ub3;

  float err = abs(netOut - groundTruth); // Absolute Error [0.0, ~2.0]
  float normErr = clamp(err * 0.5, 0.0, 1.0);

  // Hot error colormap: Black -> Indigo -> Red -> Yellow -> White
  vec3 col = vec3(0.0);
  if (normErr < 0.25) {
    col = mix(vec3(0.06, 0.08, 0.16), vec3(0.24, 0.12, 0.45), normErr * 4.0);
  } else if (normErr < 0.5) {
    col = mix(vec3(0.24, 0.12, 0.45), vec3(0.85, 0.18, 0.22), (normErr - 0.25) * 4.0);
  } else if (normErr < 0.75) {
    col = mix(vec3(0.85, 0.18, 0.22), vec3(0.98, 0.72, 0.12), (normErr - 0.5) * 4.0);
  } else {
    col = mix(vec3(0.98, 0.72, 0.12), vec3(1.0, 1.0, 1.0), (normErr - 0.75) * 4.0);
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

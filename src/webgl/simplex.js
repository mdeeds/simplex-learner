/**
 * Exact Stefan Gustavson / Ian McEwan (Ashima Arts) 2D Simplex Noise in JavaScript
 * Bit-for-bit mathematical equivalent of the standard WebGL GLSL snoise shader.
 * Range: [-1.0, 1.0]
 */

function mod289_3(x, y, z) {
  return [
    x - Math.floor(x * (1.0 / 289.0)) * 289.0,
    y - Math.floor(y * (1.0 / 289.0)) * 289.0,
    z - Math.floor(z * (1.0 / 289.0)) * 289.0
  ];
}

function permute_3(x, y, z) {
  return mod289_3(
    ((x * 34.0) + 1.0) * x,
    ((y * 34.0) + 1.0) * y,
    ((z * 34.0) + 1.0) * z
  );
}

function fract(x) {
  return x - Math.floor(x);
}

/**
 * 2D Simplex Noise
 * @param {number} xin - X coordinate float
 * @param {number} yin - Y coordinate float
 * @returns {number} Value strictly in [-1.0, 1.0]
 */
export function simplex2D(xin, yin) {
  const C_x = 0.211324865405187; // (3.0 - sqrt(3.0)) / 6.0
  const C_y = 0.366025403784439; // 0.5 * (sqrt(3.0) - 1.0)
  const C_z = -0.577350269189626; // -1.0 + 2.0 * C.x
  const C_w = 0.024390243902439; // 1.0 / 41.0

  // Skew input space
  const s = (xin + yin) * C_y;
  const i0 = Math.floor(xin + s);
  const i1 = Math.floor(yin + s);

  const t = (i0 + i1) * C_x;
  const x0_x = xin - i0 + t;
  const x0_y = yin - i1 + t;

  const i1_x = (x0_x > x0_y) ? 1.0 : 0.0;
  const i1_y = (x0_x > x0_y) ? 0.0 : 1.0;

  const x12_x = x0_x + C_x - i1_x;
  const x12_y = x0_y + C_x - i1_y;
  const x12_z = x0_x + C_z;
  const x12_w = x0_y + C_z;

  const i_x = i0 - Math.floor(i0 * (1.0 / 289.0)) * 289.0;
  const i_y = i1 - Math.floor(i1 * (1.0 / 289.0)) * 289.0;

  // p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0))
  const [py0, py1, py2] = permute_3(i_y + 0.0, i_y + i1_y, i_y + 1.0);
  const [p0, p1, p2] = permute_3(py0 + i_x + 0.0, py1 + i_x + i1_x, py2 + i_x + 1.0);

  let m0 = Math.max(0.5 - (x0_x * x0_x + x0_y * x0_y), 0.0);
  let m1 = Math.max(0.5 - (x12_x * x12_x + x12_y * x12_y), 0.0);
  let m2 = Math.max(0.5 - (x12_z * x12_z + x12_w * x12_w), 0.0);

  m0 = m0 * m0; m0 = m0 * m0;
  m1 = m1 * m1; m1 = m1 * m1;
  m2 = m2 * m2; m2 = m2 * m2;

  const x_x = 2.0 * fract(p0 * C_w) - 1.0;
  const x_y = 2.0 * fract(p1 * C_w) - 1.0;
  const x_z = 2.0 * fract(p2 * C_w) - 1.0;

  const h_x = Math.abs(x_x) - 0.5;
  const h_y = Math.abs(x_y) - 0.5;
  const h_z = Math.abs(x_z) - 0.5;

  const ox_x = Math.floor(x_x + 0.5);
  const ox_y = Math.floor(x_y + 0.5);
  const ox_z = Math.floor(x_z + 0.5);

  const a0_x = x_x - ox_x;
  const a0_y = x_y - ox_y;
  const a0_z = x_z - ox_z;

  m0 *= (1.79284291400159 - 0.85373472095314 * (a0_x * a0_x + h_x * h_x));
  m1 *= (1.79284291400159 - 0.85373472095314 * (a0_y * a0_y + h_y * h_y));
  m2 *= (1.79284291400159 - 0.85373472095314 * (a0_z * a0_z + h_z * h_z));

  const g0 = a0_x * x0_x + h_x * x0_y;
  const g1 = a0_y * x12_x + h_y * x12_y;
  const g2 = a0_z * x12_z + h_z * x12_w;

  return 130.0 * (m0 * g0 + m1 * g1 + m2 * g2);
}

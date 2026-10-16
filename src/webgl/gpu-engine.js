/**
 * WebGL GPU Math Neural Network Engine
 * 3-Layer Neural Network approximating 2D Simplex Noise
 * Input: Exactly TWO floating point numbers (x, y)
 * Layer 1: 2 -> 16
 * Layer 2: 16 -> 16
 * Layer 3: 16 -> 1
 * Total parameters: 337
 */

import {
  VERTEX_SHADER,
  FRAGMENT_SHADER_SIMPLEX,
  FRAGMENT_SHADER_NEURAL_NET,
  FRAGMENT_SHADER_DIFFERENCE
} from './shaders.js';
import { simplex2D } from './simplex.js';

export class GPUNeuralNetEngine {
  constructor(options = {}) {
    this.frequency = options.frequency ?? 2.0;
    this.activation = options.activation ?? 1; // 1=Sine (SIREN - fast coordinate fit), 0=Tanh, 2=Sigmoid
    this.omega = options.omega ?? 2.5;
    this.colormap = options.colormap ?? 1; // 0=Grayscale, 1=Turbo, 2=Viridis, 3=Cool-Warm, 4=Plasma
    this.contours = options.contours ? 1.0 : 0.0;
    this.learningRate = options.learningRate ?? 0.015;
    this.optimizer = options.optimizer ?? 'adam'; // 'sgd', 'momentum', 'adam'
    this.momentumBeta = 0.9;
    this.adamBeta1 = 0.9;
    this.adamBeta2 = 0.999;
    this.adamEps = 1e-8;
    this.batchGridSize = options.batchGridSize ?? 32; // 32x32 = 1024 sample points

    this.stepCount = 0;
    this.currentLoss = 1.0;
    this.lossHistory = [];

    // Layer dimensions
    this.H1 = 16;
    this.H2 = 16;
    this.totalParams = 2 * 16 + 16 + 16 * 16 + 16 + 16 + 1; // 337

    // Weight allocations
    this.W1 = new Float32Array(this.H1 * 2);  // 32
    this.b1 = new Float32Array(this.H1);      // 16
    this.W2 = new Float32Array(this.H2 * this.H1); // 256
    this.b2 = new Float32Array(this.H2);      // 16
    this.W3 = new Float32Array(this.H2);      // 16
    this.b3 = 0.0;                           // 1

    // Pre-allocated SIMD buffers for WebGL upload
    this.w2_0 = new Float32Array(64);
    this.w2_1 = new Float32Array(64);
    this.w2_2 = new Float32Array(64);
    this.w2_3 = new Float32Array(64);

    // Optimizer states (momentum & adam)
    this.vW1 = new Float32Array(this.W1.length);
    this.vb1 = new Float32Array(this.b1.length);
    this.vW2 = new Float32Array(this.W2.length);
    this.vb2 = new Float32Array(this.b2.length);
    this.vW3 = new Float32Array(this.W3.length);
    this.vb3 = 0.0;

    this.mW1 = new Float32Array(this.W1.length);
    this.mb1 = new Float32Array(this.b1.length);
    this.mW2 = new Float32Array(this.W2.length);
    this.mb2 = new Float32Array(this.b2.length);
    this.mW3 = new Float32Array(this.W3.length);
    this.mb3 = 0.0;

    this.sW1 = new Float32Array(this.W1.length);
    this.sb1 = new Float32Array(this.b1.length);
    this.sW2 = new Float32Array(this.W2.length);
    this.sb2 = new Float32Array(this.b2.length);
    this.sW3 = new Float32Array(this.W3.length);
    this.sb3 = 0.0;

    // WebGL context caches
    this.glContexts = new WeakMap();

    // Initialize grid and random weights
    this.precomputeSampleGrid();
    this.randomizeWeights();
  }

  /**
   * Prepares SIMD 4-float vectors for Layer 2 GPU uniforms
   */
  prepareW2Buffers() {
    for (let j = 0; j < 16; j++) {
      const rowOffset = j * 16;
      const vecOffset = j * 4;
      this.w2_0[vecOffset]     = this.W2[rowOffset];
      this.w2_0[vecOffset + 1] = this.W2[rowOffset + 1];
      this.w2_0[vecOffset + 2] = this.W2[rowOffset + 2];
      this.w2_0[vecOffset + 3] = this.W2[rowOffset + 3];

      this.w2_1[vecOffset]     = this.W2[rowOffset + 4];
      this.w2_1[vecOffset + 1] = this.W2[rowOffset + 5];
      this.w2_1[vecOffset + 2] = this.W2[rowOffset + 6];
      this.w2_1[vecOffset + 3] = this.W2[rowOffset + 7];

      this.w2_2[vecOffset]     = this.W2[rowOffset + 8];
      this.w2_2[vecOffset + 1] = this.W2[rowOffset + 9];
      this.w2_2[vecOffset + 2] = this.W2[rowOffset + 10];
      this.w2_2[vecOffset + 3] = this.W2[rowOffset + 11];

      this.w2_3[vecOffset]     = this.W2[rowOffset + 12];
      this.w2_3[vecOffset + 1] = this.W2[rowOffset + 13];
      this.w2_3[vecOffset + 2] = this.W2[rowOffset + 14];
      this.w2_3[vecOffset + 3] = this.W2[rowOffset + 15];
    }
  }

  /**
   * Randomly initializes weight matrices and biases
   */
  randomizeWeights() {
    this.stepCount = 0;
    this.lossHistory = [];

    // Xavier / Glorot initialization
    const std1 = Math.sqrt(2.0 / (2.0 + this.H1));
    for (let i = 0; i < this.W1.length; i++) {
      this.W1[i] = (Math.random() * 2.0 - 1.0) * std1 * 1.5;
    }
    for (let i = 0; i < this.b1.length; i++) {
      this.b1[i] = (Math.random() * 2.0 - 1.0) * 0.1;
    }

    const std2 = Math.sqrt(2.0 / (this.H1 + this.H2));
    for (let i = 0; i < this.W2.length; i++) {
      this.W2[i] = (Math.random() * 2.0 - 1.0) * std2;
    }
    for (let i = 0; i < this.b2.length; i++) {
      this.b2[i] = 0.0;
    }

    const std3 = Math.sqrt(2.0 / (this.H2 + 1.0));
    for (let i = 0; i < this.W3.length; i++) {
      this.W3[i] = (Math.random() * 2.0 - 1.0) * std3;
    }
    this.b3 = 0.0;

    // Reset optimizer accumulators
    this.vW1.fill(0); this.vb1.fill(0);
    this.vW2.fill(0); this.vb2.fill(0);
    this.vW3.fill(0); this.vb3 = 0;

    this.mW1.fill(0); this.mb1.fill(0);
    this.mW2.fill(0); this.mb2.fill(0);
    this.mW3.fill(0); this.mb3 = 0;

    this.sW1.fill(0); this.sb1.fill(0);
    this.sW2.fill(0); this.sb2.fill(0);
    this.sW3.fill(0); this.sb3 = 0;

    this.prepareW2Buffers();

    // Calculate initial loss
    this.currentLoss = this.evaluateLoss();
    this.lossHistory.push({ step: 0, loss: this.currentLoss });
  }

  /**
   * Precomputes training sample grid
   */
  precomputeSampleGrid() {
    const N = this.batchGridSize;
    const totalPoints = N * N;
    this.samples = new Float32Array(totalPoints * 2);
    this.targets = new Float32Array(totalPoints);

    let idx = 0;
    for (let y = 0; y < N; y++) {
      const v = ((y + 0.5) / N - 0.5) * this.frequency;
      for (let x = 0; x < N; x++) {
        const u = ((x + 0.5) / N - 0.5) * this.frequency;
        this.samples[idx * 2] = u;
        this.samples[idx * 2 + 1] = v;
        this.targets[idx] = simplex2D(u, v);
        idx++;
      }
    }
  }

  /**
   * Update frequency and recompute targets
   */
  setFrequency(freq) {
    if (this.frequency !== freq) {
      this.frequency = freq;
      this.precomputeSampleGrid();
      this.currentLoss = this.evaluateLoss();
    }
  }

  /**
   * Activation function and its derivative
   */
  act(x) {
    if (this.activation === 0) { // Tanh
      return Math.tanh(x);
    } else if (this.activation === 1) { // Sine (SIREN)
      return Math.sin(this.omega * x);
    } else { // Sigmoid
      return 1.0 / (1.0 + Math.exp(-Math.max(-12, Math.min(12, x))));
    }
  }

  dAct(x, a) {
    if (this.activation === 0) { // Tanh: 1 - a^2
      return 1.0 - a * a;
    } else if (this.activation === 1) { // Sine: omega * cos(omega * x)
      return this.omega * Math.cos(this.omega * x);
    } else { // Sigmoid: a * (1 - a)
      return a * (1.0 - a);
    }
  }

  /**
   * Forward pass: Input is strictly TWO floating-point numbers: x and y
   * Returns layer activations and scalar prediction out
   */
  forward(x, y) {
    // Layer 1: Takes exactly TWO floating point numbers: (x, y)
    const a1 = new Float32Array(this.H1);
    const z1 = new Float32Array(this.H1);
    for (let i = 0; i < this.H1; i++) {
      const z = this.W1[i * 2] * x + this.W1[i * 2 + 1] * y + this.b1[i];
      z1[i] = z;
      a1[i] = this.act(z);
    }

    // Layer 2: 16 -> 16
    const a2 = new Float32Array(this.H2);
    const z2 = new Float32Array(this.H2);
    for (let j = 0; j < this.H2; j++) {
      let sum = this.b2[j];
      const offset = j * this.H1;
      for (let i = 0; i < this.H1; i++) {
        sum += this.W2[offset + i] * a1[i];
      }
      z2[j] = sum;
      a2[j] = this.act(sum);
    }

    // Layer 3: 16 -> 1 scalar output
    let out = this.b3;
    for (let j = 0; j < this.H2; j++) {
      out += this.W3[j] * a2[j];
    }

    return { z1, a1, z2, a2, out };
  }

  /**
   * Evaluates Mean Squared Error across current sample grid
   */
  evaluateLoss() {
    if (!this.targets) return 1.0;
    const total = this.targets.length;
    let sumErrSq = 0;
    for (let k = 0; k < total; k++) {
      const x = this.samples[k * 2];
      const y = this.samples[k * 2 + 1];
      const target = this.targets[k];
      const { out } = this.forward(x, y);
      const diff = out - target;
      sumErrSq += diff * diff;
    }
    return sumErrSq / total;
  }

  /**
   * Performs ONE single gradient step
   */
  step() {
    const total = this.targets.length;
    const invTotal = 1.0 / total;

    // Gradient accumulators
    const gW1 = new Float32Array(this.W1.length);
    const gb1 = new Float32Array(this.b1.length);
    const gW2 = new Float32Array(this.W2.length);
    const gb2 = new Float32Array(this.b2.length);
    const gW3 = new Float32Array(this.W3.length);
    let gb3 = 0.0;

    let lossAcc = 0.0;

    // Vectorized backpropagation across samples
    for (let k = 0; k < total; k++) {
      const x = this.samples[k * 2];
      const y = this.samples[k * 2 + 1];
      const target = this.targets[k];

      const { z1, a1, z2, a2, out } = this.forward(x, y);
      const err = out - target;
      lossAcc += err * err;

      // Layer 3 backpropagation: Exact analytical gradient of MSE (dL/dout = 2 * err)
      const delta3 = 2.0 * err;
      gb3 += delta3;
      for (let j = 0; j < this.H2; j++) {
        gW3[j] += delta3 * a2[j];
      }

      // Layer 2 backpropagation
      const delta2 = new Float32Array(this.H2);
      for (let j = 0; j < this.H2; j++) {
        const d = (delta3 * this.W3[j]) * this.dAct(z2[j], a2[j]);
        delta2[j] = d;
        gb2[j] += d;
        const offset = j * this.H1;
        for (let i = 0; i < this.H1; i++) {
          gW2[offset + i] += d * a1[i];
        }
      }

      // Layer 1 backpropagation (Input is strictly two floating-point numbers: x and y)
      for (let i = 0; i < this.H1; i++) {
        let sum = 0.0;
        for (let j = 0; j < this.H2; j++) {
          sum += delta2[j] * this.W2[j * this.H1 + i];
        }
        const delta1 = sum * this.dAct(z1[i], a1[i]);
        gb1[i] += delta1;
        gW1[i * 2] += delta1 * x;
        gW1[i * 2 + 1] += delta1 * y;
      }
    }

    this.stepCount++;

    // Apply optimizer update
    const lr = this.learningRate;

    if (this.optimizer === 'sgd') {
      // Pure Vanilla Gradient Descent
      for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * gW1[i] * invTotal;
      for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * gb1[i] * invTotal;
      for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * gW2[i] * invTotal;
      for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * gb2[i] * invTotal;
      for (let i = 0; i < this.W3.length; i++) this.W3[i] -= lr * gW3[i] * invTotal;
      this.b3 -= lr * gb3 * invTotal;
    } else if (this.optimizer === 'momentum') {
      // Momentum SGD
      const beta = this.momentumBeta;
      for (let i = 0; i < this.W1.length; i++) {
        this.vW1[i] = beta * this.vW1[i] + (1 - beta) * (gW1[i] * invTotal);
        this.W1[i] -= lr * this.vW1[i];
      }
      for (let i = 0; i < this.b1.length; i++) {
        this.vb1[i] = beta * this.vb1[i] + (1 - beta) * (gb1[i] * invTotal);
        this.b1[i] -= lr * this.vb1[i];
      }
      for (let i = 0; i < this.W2.length; i++) {
        this.vW2[i] = beta * this.vW2[i] + (1 - beta) * (gW2[i] * invTotal);
        this.W2[i] -= lr * this.vW2[i];
      }
      for (let i = 0; i < this.b2.length; i++) {
        this.vb2[i] = beta * this.vb2[i] + (1 - beta) * (gb2[i] * invTotal);
        this.b2[i] -= lr * this.vb2[i];
      }
      for (let i = 0; i < this.W3.length; i++) {
        this.vW3[i] = beta * this.vW3[i] + (1 - beta) * (gW3[i] * invTotal);
        this.W3[i] -= lr * this.vW3[i];
      }
      this.vb3 = beta * this.vb3 + (1 - beta) * (gb3 * invTotal);
      this.b3 -= lr * this.vb3;
    } else {
      // Adam Optimizer
      const beta1 = this.adamBeta1;
      const beta2 = this.adamBeta2;
      const eps = this.adamEps;
      const t = this.stepCount;
      const fix1 = 1.0 - Math.pow(beta1, t);
      const fix2 = 1.0 - Math.pow(beta2, t);

      const updateAdam = (W, mW, sW, gVal) => {
        for (let i = 0; i < W.length; i++) {
          const g = gVal[i] * invTotal;
          mW[i] = beta1 * mW[i] + (1 - beta1) * g;
          sW[i] = beta2 * sW[i] + (1 - beta2) * g * g;
          const mHat = mW[i] / fix1;
          const sHat = sW[i] / fix2;
          W[i] -= (lr * mHat) / (Math.sqrt(sHat) + eps);
        }
      };

      updateAdam(this.W1, this.mW1, this.sW1, gW1);
      updateAdam(this.b1, this.mb1, this.sb1, gb1);
      updateAdam(this.W2, this.mW2, this.sW2, gW2);
      updateAdam(this.b2, this.mb2, this.sb2, gb2);
      updateAdam(this.W3, this.mW3, this.sW3, gW3);

      const gb3Avg = gb3 * invTotal;
      this.mb3 = beta1 * this.mb3 + (1 - beta1) * gb3Avg;
      this.sb3 = beta2 * this.sb3 + (1 - beta2) * gb3Avg * gb3Avg;
      const mHat3 = this.mb3 / fix1;
      const sHat3 = this.sb3 / fix2;
      this.b3 -= (lr * mHat3) / (Math.sqrt(sHat3) + eps);
    }

    this.prepareW2Buffers();

    // Critical fix: evaluate post-update loss so every step shows immediate progress
    this.currentLoss = this.evaluateLoss();

    this.lossHistory.push({ step: this.stepCount, loss: this.currentLoss });
    if (this.lossHistory.length > 500) {
      this.lossHistory.shift();
    }

    return {
      step: this.stepCount,
      loss: this.currentLoss
    };
  }

  /**
   * Performs N gradient steps in sequence
   * @param {number} n - Number of steps (e.g. 100)
   */
  stepN(n = 100) {
    for (let i = 0; i < n; i++) {
      this.step();
    }
    return {
      step: this.stepCount,
      loss: this.currentLoss
    };
  }

  /**
   * Compiles and caches WebGL program on canvas
   */
  getGLContext(canvas, fragmentSource) {
    let cached = this.glContexts.get(canvas);
    if (!cached || cached.fragmentSource !== fragmentSource) {
      const gl = canvas.getContext('webgl2', {
        antialias: false,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance'
      }) || canvas.getContext('webgl', {
        antialias: false,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance'
      }) || canvas.getContext('experimental-webgl');

      if (!gl) {
        throw new Error('WebGL context unavailable on this canvas');
      }

      // Compile vertex shader
      const vs = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vs, VERTEX_SHADER);
      gl.compileShader(vs);
      if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        console.error('VS Error:', gl.getShaderInfoLog(vs));
      }

      // Compile fragment shader
      const fs = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(fs, fragmentSource);
      gl.compileShader(fs);
      if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        console.error('FS Error:', gl.getShaderInfoLog(fs));
      }

      const program = gl.createProgram();
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program Link Error:', gl.getProgramInfoLog(program));
      }

      // Setup screen quad
      const quadBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      const quadVerts = new Float32Array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0
      ]);
      gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

      cached = {
        gl,
        program,
        quadBuffer,
        fragmentSource
      };
      this.glContexts.set(canvas, cached);
    }
    return cached;
  }

  /**
   * Render Canvas 1: Simplex Noise (Ground Truth)
   */
  renderSimplex(canvas, options = {}) {
    if (!canvas) return;
    try {
      const { gl, program, quadBuffer } = this.getGLContext(canvas, FRAGMENT_SHADER_SIMPLEX);

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);

      const aPos = gl.getAttribLocation(program, 'aPosition');
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const uFreq = gl.getUniformLocation(program, 'uFrequency');
      const uOff = gl.getUniformLocation(program, 'uOffset');
      const uColor = gl.getUniformLocation(program, 'uColormap');
      const uCont = gl.getUniformLocation(program, 'uContours');

      if (uFreq) gl.uniform1f(uFreq, options.frequency ?? this.frequency);
      if (uOff) gl.uniform2f(uOff, options.offsetX ?? 0.0, options.offsetY ?? 0.0);
      if (uColor) gl.uniform1i(uColor, options.colormap ?? this.colormap);
      if (uCont) gl.uniform1f(uCont, (options.contours ?? this.contours) ? 1.0 : 0.0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    } catch (err) {
      this.render2DFallback(canvas, 'simplex', options);
    }
  }

  /**
   * Render Canvas 2: Neural Network Output
   */
  renderNeuralNet(canvas, options = {}) {
    if (!canvas) return;
    try {
      const { gl, program, quadBuffer } = this.getGLContext(canvas, FRAGMENT_SHADER_NEURAL_NET);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);

    const aPos = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Standard uniforms
    const locFreq = gl.getUniformLocation(program, 'uFrequency');
    if (locFreq) gl.uniform1f(locFreq, options.frequency ?? this.frequency);
    const locOff = gl.getUniformLocation(program, 'uOffset');
    if (locOff) gl.uniform2f(locOff, options.offsetX ?? 0.0, options.offsetY ?? 0.0);
    const locColor = gl.getUniformLocation(program, 'uColormap');
    if (locColor) gl.uniform1i(locColor, options.colormap ?? this.colormap);
    const locCont = gl.getUniformLocation(program, 'uContours');
    if (locCont) gl.uniform1f(locCont, (options.contours ?? this.contours) ? 1.0 : 0.0);
    const locAct = gl.getUniformLocation(program, 'uActivation');
    if (locAct) gl.uniform1i(locAct, options.activation ?? this.activation);
    const locOmega = gl.getUniformLocation(program, 'uOmega');
    if (locOmega) gl.uniform1f(locOmega, options.omega ?? this.omega);

    // Upload Layer 1: uW1[16] (vec2), ub1[16] (float)
    for (let i = 0; i < this.H1; i++) {
      const locW = gl.getUniformLocation(program, `uW1[${i}]`);
      if (locW) gl.uniform2f(locW, this.W1[i * 2], this.W1[i * 2 + 1]);
      const locB = gl.getUniformLocation(program, `ub1[${i}]`);
      if (locB) gl.uniform1f(locB, this.b1[i]);
    }

    // Upload Layer 2: 16 neurons x 4 vec4s (64 vec4s total = 256 weights)
    for (let j = 0; j < 16; j++) {
      const o = j * 4;
      const l0 = gl.getUniformLocation(program, `uW2_0[${j}]`);
      if (l0) gl.uniform4f(l0, this.w2_0[o], this.w2_0[o + 1], this.w2_0[o + 2], this.w2_0[o + 3]);
      const l1 = gl.getUniformLocation(program, `uW2_1[${j}]`);
      if (l1) gl.uniform4f(l1, this.w2_1[o], this.w2_1[o + 1], this.w2_1[o + 2], this.w2_1[o + 3]);
      const l2 = gl.getUniformLocation(program, `uW2_2[${j}]`);
      if (l2) gl.uniform4f(l2, this.w2_2[o], this.w2_2[o + 1], this.w2_2[o + 2], this.w2_2[o + 3]);
      const l3 = gl.getUniformLocation(program, `uW2_3[${j}]`);
      if (l3) gl.uniform4f(l3, this.w2_3[o], this.w2_3[o + 1], this.w2_3[o + 2], this.w2_3[o + 3]);

      const locB2 = gl.getUniformLocation(program, `ub2[${j}]`);
      if (locB2) gl.uniform1f(locB2, this.b2[j]);
    }

    // Upload Layer 3: 4 vec4s (16 weights) + 1 bias
    const locW3_0 = gl.getUniformLocation(program, 'uW3_0');
    if (locW3_0) gl.uniform4f(locW3_0, this.W3[0], this.W3[1], this.W3[2], this.W3[3]);
    const locW3_1 = gl.getUniformLocation(program, 'uW3_1');
    if (locW3_1) gl.uniform4f(locW3_1, this.W3[4], this.W3[5], this.W3[6], this.W3[7]);
    const locW3_2 = gl.getUniformLocation(program, 'uW3_2');
    if (locW3_2) gl.uniform4f(locW3_2, this.W3[8], this.W3[9], this.W3[10], this.W3[11]);
    const locW3_3 = gl.getUniformLocation(program, 'uW3_3');
    if (locW3_3) gl.uniform4f(locW3_3, this.W3[12], this.W3[13], this.W3[14], this.W3[15]);

    const locB3 = gl.getUniformLocation(program, 'ub3');
    if (locB3) gl.uniform1f(locB3, this.b3);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  } catch (err) {
    this.render2DFallback(canvas, 'neuralnet', options);
  }
}

  /**
   * Render Canvas 3: Difference / Error Map
   */
  renderDifference(canvas, options = {}) {
    if (!canvas) return;
    try {
      const { gl, program, quadBuffer } = this.getGLContext(canvas, FRAGMENT_SHADER_DIFFERENCE);

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);

      const aPos = gl.getAttribLocation(program, 'aPosition');
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const locFreq = gl.getUniformLocation(program, 'uFrequency');
      if (locFreq) gl.uniform1f(locFreq, options.frequency ?? this.frequency);
      const locOff = gl.getUniformLocation(program, 'uOffset');
      if (locOff) gl.uniform2f(locOff, options.offsetX ?? 0.0, options.offsetY ?? 0.0);
      const locAct = gl.getUniformLocation(program, 'uActivation');
      if (locAct) gl.uniform1i(locAct, options.activation ?? this.activation);
      const locOmega = gl.getUniformLocation(program, 'uOmega');
      if (locOmega) gl.uniform1f(locOmega, options.omega ?? this.omega);

      // Layer 1
      for (let i = 0; i < this.H1; i++) {
        const locW = gl.getUniformLocation(program, `uW1[${i}]`);
        if (locW) gl.uniform2f(locW, this.W1[i * 2], this.W1[i * 2 + 1]);
        const locB = gl.getUniformLocation(program, `ub1[${i}]`);
        if (locB) gl.uniform1f(locB, this.b1[i]);
      }

      // Layer 2: Indexed upload for reliable cross-browser execution
      for (let j = 0; j < 16; j++) {
        const o = j * 4;
        const l0 = gl.getUniformLocation(program, `uW2_0[${j}]`);
        if (l0) gl.uniform4f(l0, this.w2_0[o], this.w2_0[o + 1], this.w2_0[o + 2], this.w2_0[o + 3]);
        const l1 = gl.getUniformLocation(program, `uW2_1[${j}]`);
        if (l1) gl.uniform4f(l1, this.w2_1[o], this.w2_1[o + 1], this.w2_1[o + 2], this.w2_1[o + 3]);
        const l2 = gl.getUniformLocation(program, `uW2_2[${j}]`);
        if (l2) gl.uniform4f(l2, this.w2_2[o], this.w2_2[o + 1], this.w2_2[o + 2], this.w2_2[o + 3]);
        const l3 = gl.getUniformLocation(program, `uW2_3[${j}]`);
        if (l3) gl.uniform4f(l3, this.w2_3[o], this.w2_3[o + 1], this.w2_3[o + 2], this.w2_3[o + 3]);

        const locB2 = gl.getUniformLocation(program, `ub2[${j}]`);
        if (locB2) gl.uniform1f(locB2, this.b2[j]);
      }

      // Layer 3
      const locW3_0 = gl.getUniformLocation(program, 'uW3_0');
      if (locW3_0) gl.uniform4f(locW3_0, this.W3[0], this.W3[1], this.W3[2], this.W3[3]);
      const locW3_1 = gl.getUniformLocation(program, 'uW3_1');
      if (locW3_1) gl.uniform4f(locW3_1, this.W3[4], this.W3[5], this.W3[6], this.W3[7]);
      const locW3_2 = gl.getUniformLocation(program, 'uW3_2');
      if (locW3_2) gl.uniform4f(locW3_2, this.W3[8], this.W3[9], this.W3[10], this.W3[11]);
      const locW3_3 = gl.getUniformLocation(program, 'uW3_3');
      if (locW3_3) gl.uniform4f(locW3_3, this.W3[12], this.W3[13], this.W3[14], this.W3[15]);

      const locB3 = gl.getUniformLocation(program, 'ub3');
      if (locB3) gl.uniform1f(locB3, this.b3);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    } catch (err) {
      this.render2DFallback(canvas, 'difference', options);
    }
  }

  /**
   * Ultra-fast Canvas 2D Fallback Renderer
   * Automatically invoked if WebGL context is lost, unsupported, or encounters an error.
   */
  render2DFallback(canvas, mode = 'neuralnet', options = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    const freq = options.frequency ?? this.frequency;

    let p = 0;
    for (let py = 0; py < h; py++) {
      const v = (0.5 - py / h) * freq;
      for (let px = 0; px < w; px++) {
        const u = (px / w - 0.5) * freq;
        let val = 0;
        if (mode === 'simplex') {
          val = simplex2D(u, v);
        } else if (mode === 'neuralnet') {
          val = this.forward(u, v).out;
        } else {
          const target = simplex2D(u, v);
          const pred = this.forward(u, v).out;
          val = Math.abs(pred - target);
        }

        const norm = Math.max(0, Math.min(1, val * 0.5 + 0.5));
        // Turbo color approximation
        const r = Math.floor(255 * Math.max(0, Math.min(1, 0.1357 + 4.615 * norm - 42.66 * norm * norm + 132.13 * norm * norm * norm - 152.94 * Math.pow(norm, 4) + 59.28 * Math.pow(norm, 5))));
        const g = Math.floor(255 * Math.max(0, Math.min(1, 0.0914 + 2.194 * norm + 4.842 * norm * norm - 14.185 * norm * norm * norm + 4.277 * Math.pow(norm, 4) + 2.829 * Math.pow(norm, 5))));
        const b = Math.floor(255 * Math.max(0, Math.min(1, 0.1066 + 12.64 * norm - 60.58 * norm * norm + 110.36 * norm * norm * norm - 89.90 * Math.pow(norm, 4) + 27.34 * Math.pow(norm, 5))));

        data[p] = r;
        data[p + 1] = g;
        data[p + 2] = b;
        data[p + 3] = 255;
        p += 4;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  /**
   * Probe point (u, v) normalized to [0, 1]
   * Input to forward() is strictly TWO floating-point numbers: x and y
   */
  probePoint(u, v) {
    const x = (u - 0.5) * this.frequency;
    const y = (v - 0.5) * this.frequency;
    const target = simplex2D(x, y);
    const { out } = this.forward(x, y);
    return {
      x,
      y,
      u,
      v,
      target,
      prediction: out,
      error: Math.abs(out - target)
    };
  }

  /**
   * Computes min, max, center predictions across domain
   */
  getPredictionSummary() {
    const center = this.forward(0.0, 0.0).out;
    const centerTarget = simplex2D(0.0, 0.0);

    let minPred = Infinity;
    let maxPred = -Infinity;

    // Sample across grid
    const N = 12;
    for (let j = 0; j < N; j++) {
      const v = ((j + 0.5) / N - 0.5) * this.frequency;
      for (let i = 0; i < N; i++) {
        const u = ((i + 0.5) / N - 0.5) * this.frequency;
        const p = this.forward(u, v).out;
        if (p < minPred) minPred = p;
        if (p > maxPred) maxPred = p;
      }
    }

    return {
      center: {
        x: 0.0,
        y: 0.0,
        target: centerTarget,
        pred: center,
        error: Math.abs(center - centerTarget)
      },
      minPred,
      maxPred
    };
  }

  /**
   * Weight statistics for Inspector
   */
  getWeightStats() {
    const computeStats = (arr) => {
      let min = Infinity, max = -Infinity, sum = 0, sqSum = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sqSum += v * v;
      }
      const mean = sum / arr.length;
      const std = Math.sqrt(Math.max(0, sqSum / arr.length - mean * mean));
      return { min, max, mean, std, length: arr.length };
    };

    return {
      W1: computeStats(this.W1),
      b1: computeStats(this.b1),
      W2: computeStats(this.W2),
      b2: computeStats(this.b2),
      W3: computeStats(this.W3),
      b3: this.b3,
      totalParams: this.totalParams
    };
  }
}

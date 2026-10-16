/**
 * Main Application Controller - Vanilla JavaScript with ES Modules
 * Statically hosted, zero frameworks, zero bundlers, no TypeScript.
 */

import { GPUNeuralNetEngine } from './webgl/gpu-engine.js';
import { simplex2D } from './webgl/simplex.js';

function initApp() {
  try {
    // Initialize WebGL GPU Neural Network Engine
    const engine = new GPUNeuralNetEngine({
      frequency: 2.0,
      learningRate: 0.015,
      optimizer: 'adam',
      activation: 1, // 1=Sine (SIREN) for optimal 2D signal approximation
      colormap: 1    // Turbo
    });

    // Canvas references
    const canvasSimplex = document.getElementById('canvas-simplex');
    const canvasNeuralNet = document.getElementById('canvas-neural-net');
    const canvasDiff = document.getElementById('canvas-diff');

    if (!canvasSimplex || !canvasNeuralNet || !canvasDiff) {
      console.error('Canvas elements not found');
      return;
    }

    // Metric elements
    const metricSteps = document.getElementById('metric-steps');
    const metricLoss = document.getElementById('metric-loss');
    const metricReduction = document.getElementById('metric-reduction');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    // State
    let initialLoss = engine.currentLoss;
    let isTraining = false;
    let trainingAnimId = null;
    let autoSpeed = 5;
    let useLogScale = false;
    let selectedWeightLayer = 'W2';
    let currentSliceY = 0.0;

    /**
     * Re-render all three canvases
     */
    function renderCanvases() {
      try {
        engine.renderSimplex(canvasSimplex);
        engine.renderNeuralNet(canvasNeuralNet);
        engine.renderDifference(canvasDiff);
      } catch (err) {
        console.warn('Canvas render issue, using fallback:', err);
      }
    }

    /**
     * Update UI metrics (Steps, MSE Loss, Reduction %)
     */
    function updateMetrics() {
      if (metricSteps) metricSteps.textContent = engine.stepCount.toLocaleString();
      if (metricLoss) metricLoss.textContent = engine.currentLoss.toFixed(5);

      if (metricReduction && initialLoss > 0) {
        const pct = (((initialLoss - engine.currentLoss) / initialLoss) * 100).toFixed(1);
        metricReduction.textContent = pct >= 0 ? `-${pct}%` : `+${Math.abs(pct)}%`;
      }
    }

    /**
     * Render SVG Loss History Chart
     */
    function renderLossChart() {
      const pathArea = document.getElementById('chart-area');
      const pathLine = document.getElementById('chart-line');
      const minLossElem = document.getElementById('chart-min-loss');

      const history = engine.lossHistory;
      if (!history || history.length < 2 || !pathArea || !pathLine) return;

      const width = 600;
      const height = 150;
      const pad = { top: 15, right: 15, bottom: 25, left: 55 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;

      const losses = history.map((h) => h.loss);
      const minL = Math.min(...losses);
      const maxL = Math.max(...losses);

      if (minLossElem) {
        minLossElem.textContent = `Min: ${minL.toFixed(5)}`;
      }

      const getY = (val) => {
        if (useLogScale) {
          const minLog = Math.log10(Math.max(1e-4, minL * 0.8));
          const maxLog = Math.log10(Math.max(1e-3, maxL * 1.1));
          const valLog = Math.log10(Math.max(1e-4, val));
          const norm = (valLog - minLog) / (maxLog - minLog || 1);
          return pad.top + plotH * (1.0 - Math.max(0, Math.min(1, norm)));
        } else {
          const range = maxL - minL || 0.01;
          const norm = (val - minL) / range;
          return pad.top + plotH * (1.0 - norm);
        }
      };

      const getX = (idx) => {
        return pad.left + (idx / (history.length - 1)) * plotW;
      };

      let lineD = `M ${getX(0)} ${getY(history[0].loss)}`;
      for (let i = 1; i < history.length; i++) {
        lineD += ` L ${getX(i)} ${getY(history[i].loss)}`;
      }

      const areaD = `${lineD} L ${getX(history.length - 1)} ${pad.top + plotH} L ${getX(0)} ${pad.top + plotH} Z`;

      pathArea.setAttribute('d', areaD);
      pathLine.setAttribute('d', lineD);
    }

    /**
     * Render 1D Horizontal Slice Profile
     */
    function renderSliceProfile() {
      const targetPathElem = document.getElementById('slice-target-line');
      const netPathElem = document.getElementById('slice-net-line');
      if (!targetPathElem || !netPathElem) return;

      const points = 100;
      const width = 600;
      const height = 140;
      const pad = { top: 15, right: 15, bottom: 25, left: 40 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;

      const yCoord = currentSliceY * engine.frequency * 0.5;

      const getY = (val) => {
        const clamped = Math.max(-1.5, Math.min(1.5, val));
        const norm = (clamped - (-1.5)) / 3.0; // 0 to 1
        return pad.top + plotH * (1.0 - norm);
      };

      const getX = (idx) => {
        return pad.left + (idx / (points - 1)) * plotW;
      };

      let targetD = '';
      let netD = '';

      for (let i = 0; i < points; i++) {
        const u = i / (points - 1);
        const x = (u - 0.5) * engine.frequency;
        const targetVal = simplex2D(x, yCoord);
        const { out } = engine.forward(x, yCoord);

        if (i === 0) {
          targetD = `M ${getX(0)} ${getY(targetVal)}`;
          netD = `M ${getX(0)} ${getY(out)}`;
        } else {
          targetD += ` L ${getX(i)} ${getY(targetVal)}`;
          netD += ` L ${getX(i)} ${getY(out)}`;
        }
      }

      targetPathElem.setAttribute('d', targetD);
      netPathElem.setAttribute('d', netD);
    }

    /**
     * Render Weight Matrix Heatmap
     */
    function renderWeightInspector() {
      const display = document.getElementById('matrix-display');
      const statLayer = document.getElementById('stat-layer');
      const statMin = document.getElementById('stat-min');
      const statMax = document.getElementById('stat-max');
      const statStd = document.getElementById('stat-std');
      const hoverInfo = document.getElementById('matrix-hover-info');

      if (!display) return;

      const stats = engine.getWeightStats();
      const currentStat = stats[selectedWeightLayer];

      if (statLayer) statLayer.textContent = selectedWeightLayer;
      if (currentStat && currentStat.min !== undefined) {
        if (statMin) statMin.textContent = currentStat.min.toFixed(4);
        if (statMax) statMax.textContent = currentStat.max.toFixed(4);
        if (statStd) statStd.textContent = currentStat.std.toFixed(4);
      } else {
        if (statMin) statMin.textContent = engine.b3.toFixed(4);
        if (statMax) statMax.textContent = engine.b3.toFixed(4);
        if (statStd) statStd.textContent = '0.0000';
      }

      display.innerHTML = '';

      const getCellColor = (val) => {
        const norm = Math.max(-1.5, Math.min(1.5, val)) / 1.5;
        if (norm < 0) {
          const alpha = 0.2 + Math.min(1, Math.abs(norm)) * 0.75;
          return `rgba(59, 130, 246, ${alpha})`;
        } else {
          const alpha = 0.2 + Math.min(1, norm) * 0.75;
          return `rgba(244, 63, 94, ${alpha})`;
        }
      };

      if (selectedWeightLayer === 'W1') {
        // 16 neurons x 2 inputs
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(2, 1fr)';
        grid.style.gap = '4px';
        grid.style.maxWidth = '280px';
        grid.style.width = '100%';

        for (let i = 0; i < 16; i++) {
          for (let c = 0; c < 2; c++) {
            const val = engine.W1[i * 2 + c];
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.height = '24px';
            cell.style.borderRadius = '4px';
            cell.style.background = getCellColor(val);
            cell.style.display = 'flex';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'space-between';
            cell.style.padding = '0 6px';
            cell.style.fontSize = '9px';
            cell.style.fontFamily = 'monospace';
            cell.style.color = '#ffffff';
            cell.innerHTML = `<span>N${i}.${c === 0 ? 'x' : 'y'}</span><span>${val.toFixed(2)}</span>`;

            cell.onmouseenter = () => {
              if (hoverInfo) hoverInfo.textContent = `W1[neuron: ${i}, input: ${c === 0 ? 'X' : 'Y'}] = ${val.toFixed(5)}`;
            };
            cell.onmouseleave = () => {
              if (hoverInfo) hoverInfo.textContent = 'Hover over any weight cell to inspect value';
            };

            grid.appendChild(cell);
          }
        }
        display.appendChild(grid);
      } else if (selectedWeightLayer === 'W2') {
        // 16x16 matrix (256 weights)
        const grid = document.createElement('div');
        grid.className = 'matrix-grid-16';

        for (let idx = 0; idx < 256; idx++) {
          const row = Math.floor(idx / 16);
          const col = idx % 16;
          const val = engine.W2[idx];

          const cell = document.createElement('div');
          cell.className = 'matrix-cell';
          cell.style.background = getCellColor(val);

          cell.onmouseenter = () => {
            if (hoverInfo) hoverInfo.textContent = `W2[row: ${row}, col: ${col}] = ${val.toFixed(5)}`;
          };
          cell.onmouseleave = () => {
            if (hoverInfo) hoverInfo.textContent = 'Hover over any weight cell to inspect value';
          };

          grid.appendChild(cell);
        }
        display.appendChild(grid);
      } else if (selectedWeightLayer === 'W3') {
        // 1x16 weights
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(16, 1fr)';
        grid.style.gap = '4px';
        grid.style.maxWidth = '440px';
        grid.style.width = '100%';

        for (let j = 0; j < 16; j++) {
          const val = engine.W3[j];
          const cell = document.createElement('div');
          cell.className = 'matrix-cell';
          cell.style.height = '40px';
          cell.style.borderRadius = '4px';
          cell.style.background = getCellColor(val);
          cell.style.display = 'flex';
          cell.style.flexDirection = 'column';
          cell.style.alignItems = 'center';
          cell.style.justifyContent = 'center';
          cell.style.fontSize = '9px';
          cell.style.fontFamily = 'monospace';
          cell.style.color = '#ffffff';
          cell.innerHTML = `<span>j${j}</span><span>${val.toFixed(2)}</span>`;

          cell.onmouseenter = () => {
            if (hoverInfo) hoverInfo.textContent = `W3[output, neuron ${j}] = ${val.toFixed(5)} (bias b3 = ${engine.b3.toFixed(4)})`;
          };
          cell.onmouseleave = () => {
            if (hoverInfo) hoverInfo.textContent = 'Hover over any weight cell to inspect value';
          };

          grid.appendChild(cell);
        }
        display.appendChild(grid);
      }
    }

    let currentProbedU = 0.5;
    let currentProbedV = 0.5;

    /**
     * Update live prediction monitor and Canvas 2 footer readouts
     */
    function updatePredictionMonitor(u = currentProbedU, v = currentProbedV) {
      currentProbedU = u;
      currentProbedV = v;

      const probe = engine.probePoint(u, v);
      const summary = engine.getPredictionSummary();

      const monitorInput = document.getElementById('monitor-input');
      const monitorTarget = document.getElementById('monitor-target');
      const monitorPred = document.getElementById('monitor-pred');
      const monitorErr = document.getElementById('monitor-err');
      const monitorRange = document.getElementById('monitor-range');

      if (monitorInput) monitorInput.textContent = `(${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})`;
      if (monitorTarget) monitorTarget.textContent = (probe.target >= 0 ? '+' : '') + probe.target.toFixed(4);
      if (monitorPred) monitorPred.textContent = (probe.prediction >= 0 ? '+' : '') + probe.prediction.toFixed(4);
      if (monitorErr) monitorErr.textContent = probe.error.toFixed(4);
      if (monitorRange) monitorRange.textContent = `[${summary.minPred.toFixed(2)}, ${summary.maxPred.toFixed(2)}]`;

      const probeCoords = document.getElementById('probe-pred-coords');
      const probeVal = document.getElementById('probe-pred-val');
      if (probeCoords) probeCoords.textContent = `coord: (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})`;
      if (probeVal) probeVal.innerHTML = `Pred: <b>${(probe.prediction >= 0 ? '+' : '') + probe.prediction.toFixed(4)}</b> &bull; Target: ${probe.target.toFixed(4)}`;
    }

    /**
     * Execute a single gradient step
     */
    function runSingleStep() {
      engine.step();
      updateMetrics();
      renderCanvases();
      updatePredictionMonitor();
      renderLossChart();
      renderSliceProfile();
      renderWeightInspector();
    }

    /**
     * Execute 100 gradient steps
     */
    function runHundredSteps() {
      engine.stepN(100);
      updateMetrics();
      renderCanvases();
      updatePredictionMonitor();
      renderLossChart();
      renderSliceProfile();
      renderWeightInspector();
    }

    /**
     * Auto-train loop
     */
    function startAutoTrain() {
      isTraining = true;
      if (statusDot) statusDot.classList.add('active');
      if (statusText) statusText.textContent = 'Training...';
      const textElem = document.getElementById('text-auto-train');
      if (textElem) textElem.textContent = 'Pause';
      const btnElem = document.getElementById('btn-auto-train');
      if (btnElem) btnElem.className = 'btn-warning';

      const loop = () => {
        if (!isTraining) return;
        engine.stepN(autoSpeed);
        updateMetrics();
        renderCanvases();
        updatePredictionMonitor();
        renderLossChart();
        renderSliceProfile();
        renderWeightInspector();
        trainingAnimId = requestAnimationFrame(loop);
      };
      trainingAnimId = requestAnimationFrame(loop);
    }

    function stopAutoTrain() {
      isTraining = false;
      if (trainingAnimId) {
        cancelAnimationFrame(trainingAnimId);
        trainingAnimId = null;
      }
      if (statusDot) statusDot.classList.remove('active');
      if (statusText) statusText.textContent = 'Ready';
      const textElem = document.getElementById('text-auto-train');
      if (textElem) textElem.textContent = 'Auto Train';
      const btnElem = document.getElementById('btn-auto-train');
      if (btnElem) btnElem.className = 'btn-dark';
    }

    function toggleAutoTrain() {
      if (isTraining) {
        stopAutoTrain();
      } else {
        startAutoTrain();
      }
    }

    /**
     * Randomize weights
     */
    function randomizeWeights() {
      stopAutoTrain();
      engine.randomizeWeights();
      initialLoss = engine.currentLoss;
      updateMetrics();
      renderCanvases();
      updatePredictionMonitor();
      renderLossChart();
      renderSliceProfile();
      renderWeightInspector();
    }

    /**
     * Setup Canvas Probing and Crosshairs
     */
    function setupCanvasProbe(canvas, crosshairH, crosshairV, footerElem, type) {
      if (!canvas || !footerElem) return;
      const container = canvas.parentElement;
      if (!container) return;

      container.addEventListener('mousemove', (e) => {
        const rect = container.getBoundingClientRect();
        const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const v = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

        const xPos = e.clientX - rect.left;
        const yPos = e.clientY - rect.top;

        if (crosshairH) {
          crosshairH.style.display = 'block';
          crosshairH.style.top = `${yPos}px`;
        }
        if (crosshairV) {
          crosshairV.style.display = 'block';
          crosshairV.style.left = `${xPos}px`;
        }

        updatePredictionMonitor(u, v);
        const probe = engine.probePoint(u, v);
        if (type === 'simplex') {
          footerElem.innerHTML = `<span>coord: (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})</span><span>target: <b>${probe.target.toFixed(4)}</b></span>`;
        } else if (type === 'net') {
          footerElem.innerHTML = `<span>coord: (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})</span><span>pred: <b>${probe.prediction.toFixed(4)}</b></span>`;
        } else {
          footerElem.innerHTML = `<span>coord: (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})</span><span>error: <b>${probe.error.toFixed(4)}</b></span>`;
        }
      });

      container.addEventListener('mouseleave', () => {
        if (crosshairH) crosshairH.style.display = 'none';
        if (crosshairV) crosshairV.style.display = 'none';
        updatePredictionMonitor(0.5, 0.5);
      });
    }

    // Attach Event Listeners
    const btnSingle = document.getElementById('btn-single-step');
    if (btnSingle) btnSingle.addEventListener('click', runSingleStep);

    const btnHundred = document.getElementById('btn-hundred-steps');
    if (btnHundred) btnHundred.addEventListener('click', runHundredSteps);

    const btnAuto = document.getElementById('btn-auto-train');
    if (btnAuto) btnAuto.addEventListener('click', toggleAutoTrain);

    const btnRand = document.getElementById('btn-randomize');
    if (btnRand) btnRand.addEventListener('click', randomizeWeights);

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        runSingleStep();
      } else if (e.code === 'Enter') {
        e.preventDefault();
        runHundredSteps();
      }
    });

    // Learning Rate slider & presets
    const inputLR = document.getElementById('input-lr');
    const valLR = document.getElementById('val-lr');
    if (inputLR) {
      inputLR.addEventListener('input', (e) => {
        const lr = parseFloat(e.target.value);
        engine.learningRate = lr;
        if (valLR) valLR.textContent = lr.toFixed(3);
        document.querySelectorAll('.lr-preset').forEach((btn) => {
          btn.classList.toggle('active', Math.abs(parseFloat(btn.dataset.lr) - lr) < 0.001);
        });
      });
    }

    document.querySelectorAll('.lr-preset').forEach((btn) => {
      btn.addEventListener('click', () => {
        const lr = parseFloat(btn.dataset.lr);
        engine.learningRate = lr;
        if (inputLR) inputLR.value = lr;
        if (valLR) valLR.textContent = lr.toFixed(3);
        document.querySelectorAll('.lr-preset').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Optimizer mode buttons
    document.querySelectorAll('.opt-mode').forEach((btn) => {
      btn.addEventListener('click', () => {
        engine.optimizer = btn.dataset.opt;
        const optLabel = document.getElementById('val-opt');
        if (optLabel) optLabel.textContent = btn.textContent;
        document.querySelectorAll('.opt-mode').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Frequency slider & presets
    const inputFreq = document.getElementById('input-freq');
    const valFreq = document.getElementById('val-freq');
    if (inputFreq) {
      inputFreq.addEventListener('input', (e) => {
        const f = parseFloat(e.target.value);
        engine.setFrequency(f);
        if (valFreq) valFreq.textContent = `${f.toFixed(2)}x`;
        renderCanvases();
        updateMetrics();
        renderSliceProfile();
        document.querySelectorAll('.freq-preset').forEach((btn) => {
          btn.classList.toggle('active', Math.abs(parseFloat(btn.dataset.freq) - f) < 0.1);
        });
      });
    }

    document.querySelectorAll('.freq-preset').forEach((btn) => {
      btn.addEventListener('click', () => {
        const f = parseFloat(btn.dataset.freq);
        engine.setFrequency(f);
        if (inputFreq) inputFreq.value = f;
        if (valFreq) valFreq.textContent = `${f.toFixed(2)}x`;
        renderCanvases();
        updateMetrics();
        renderSliceProfile();
        document.querySelectorAll('.freq-preset').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Activation buttons
    document.querySelectorAll('.act-mode').forEach((btn) => {
      btn.addEventListener('click', () => {
        const act = parseInt(btn.dataset.act, 10);
        engine.activation = act;
        const actLabel = document.getElementById('val-act');
        if (actLabel) actLabel.textContent = btn.textContent;
        document.querySelectorAll('.act-mode').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        engine.currentLoss = engine.evaluateLoss();
        updateMetrics();
        renderCanvases();
        renderSliceProfile();
      });
    });

    // Colormap buttons
    document.querySelectorAll('.cmap-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const cmap = parseInt(btn.dataset.cmap, 10);
        engine.colormap = cmap;
        document.querySelectorAll('.cmap-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        renderCanvases();
      });
    });

    // Contours checkbox
    const checkContours = document.getElementById('check-contours');
    if (checkContours) {
      checkContours.addEventListener('change', (e) => {
        engine.contours = e.target.checked ? 1.0 : 0.0;
        renderCanvases();
      });
    }

    // Speed select
    const selectSpeed = document.getElementById('select-speed');
    if (selectSpeed) {
      selectSpeed.addEventListener('change', (e) => {
        autoSpeed = parseInt(e.target.value, 10);
      });
    }

    // Log scale toggle
    const btnToggleLog = document.getElementById('btn-toggle-log');
    if (btnToggleLog) {
      btnToggleLog.addEventListener('click', (e) => {
        useLogScale = !useLogScale;
        e.target.classList.toggle('active', useLogScale);
        e.target.textContent = useLogScale ? 'Linear Scale' : 'Log Scale';
        renderLossChart();
      });
    }

    // 1D slice y slider
    const inputSliceY = document.getElementById('input-slice-y');
    const valSliceY = document.getElementById('val-slice-y');
    if (inputSliceY) {
      inputSliceY.addEventListener('input', (e) => {
        currentSliceY = parseFloat(e.target.value);
        if (valSliceY) valSliceY.textContent = currentSliceY.toFixed(2);
        renderSliceProfile();
      });
    }

    // Weight layer tabs
    document.querySelectorAll('.weight-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        selectedWeightLayer = btn.dataset.layer;
        document.querySelectorAll('.weight-tab').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        renderWeightInspector();
      });
    });

    // Modal open/close
    const mathModal = document.getElementById('math-modal');
    const btnOpenMath = document.getElementById('btn-open-math');
    const btnCloseMath = document.getElementById('btn-close-math');
    if (btnOpenMath && mathModal) {
      btnOpenMath.addEventListener('click', () => {
        mathModal.style.display = 'flex';
      });
    }
    if (btnCloseMath && mathModal) {
      btnCloseMath.addEventListener('click', () => {
        mathModal.style.display = 'none';
      });
    }
    if (mathModal) {
      mathModal.addEventListener('click', (e) => {
        if (e.target === mathModal) {
          mathModal.style.display = 'none';
        }
      });
    }

    // Setup Probes for all 3 Canvases
    setupCanvasProbe(
      canvasSimplex,
      document.getElementById('crosshair-h-1'),
      document.getElementById('crosshair-v-1'),
      document.getElementById('probe-simplex'),
      'simplex'
    );
    setupCanvasProbe(
      canvasNeuralNet,
      document.getElementById('crosshair-h-2'),
      document.getElementById('crosshair-v-2'),
      document.getElementById('probe-neural-net'),
      'net'
    );
    setupCanvasProbe(
      canvasDiff,
      document.getElementById('crosshair-h-3'),
      document.getElementById('crosshair-v-3'),
      document.getElementById('probe-diff'),
      'diff'
    );

    // Initial Render
    renderCanvases();
    updateMetrics();
    updatePredictionMonitor();
    renderLossChart();
    renderSliceProfile();
    renderWeightInspector();

    console.log('WebGL Simplex Neural Net initialized successfully.');
  } catch (err) {
    console.error('Fatal initialization error:', err);
    const main = document.querySelector('main');
    if (main) {
      const banner = document.createElement('div');
      banner.style.cssText = 'background:#450a0a; color:#fca5a5; padding:1rem; border-radius:12px; margin:1rem; border:1px solid #b91c1c; font-family:monospace;';
      banner.innerHTML = `<strong>Initialization error:</strong> ${err.message}`;
      main.prepend(banner);
    }
  }
}

// Execute when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

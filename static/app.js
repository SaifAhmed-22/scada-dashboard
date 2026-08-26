// ============================================================
// SCADA Pipeline Risk Dashboard — frontend logic
// No framework: plain DOM + fetch + Chart.js (loaded via CDN in
// index.html). Kept deliberately simple so it's easy to read and
// modify alongside the Python model.
// ============================================================

const ARC_LENGTH = 251.33; // semicircle length, radius 80 (pi * r)

const state = {
  meta: null,
  segments: [],
  selectedSegment: null,
  simData: [],
  tickIndex: 0,
  playing: false,
  playTimer: null,
  chart: null,
};

// ---------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------
function alertClass(level) {
  if (level === "Critical Leak Threat") return "critical";
  if (level === "Warning") return "warning";
  return "normal";
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

const VALVE_NAMES = { 0: "CLOSED", 1: "OPEN", 2: "PARTIAL" };
const VALVE_COLORS = { 0: "#e2483d", 1: "#45b880", 2: "#e8a33d" };

// ---------------------------------------------------------------
// Boot
// ---------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  startClock();
  setupTabs();
  setupPlaybackControls();
  setupWhatIfForm();
  loadMeta();
  loadSegments();
});

function startClock() {
  const el = document.getElementById("clock");
  const tick = () => { el.textContent = new Date().toLocaleTimeString(); };
  tick();
  setInterval(tick, 1000);
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("view-" + btn.dataset.view).classList.add("active");
      if (btn.dataset.view === "monitor" && state.chart) {
        state.chart.resize();
      }
    });
  });
}

// ---------------------------------------------------------------
// Meta + Model Performance view
// ---------------------------------------------------------------
async function loadMeta() {
  const res = await fetch("/api/meta");
  const meta = await res.json();
  state.meta = meta;

  document.getElementById("stat-segments").textContent = meta.dataset.segments;
  document.getElementById("stat-auc").textContent =
    meta.model.holdout_auc !== null ? meta.model.holdout_auc.toFixed(3) : "n/a";

  renderModelView(meta);
}

function renderModelView(meta) {
  const grid = document.getElementById("metrics-grid");
  const tiles = [
    {
      label: "Holdout AUC",
      value: meta.model.holdout_auc !== null ? meta.model.holdout_auc.toFixed(3) : "—",
      note: `${meta.model.holdout_train_size} train / ${meta.model.holdout_test_size} test (chronological)`,
    },
    {
      label: "TimeSeriesSplit F1",
      value: meta.model.cv_f1_mean !== null ? meta.model.cv_f1_mean.toFixed(3) : "—",
      note: `${meta.model.cv_folds} chronological folds`,
    },
    {
      label: "Anomaly-only AUC",
      value: meta.model.anomaly_only_auc !== null ? meta.model.anomaly_only_auc.toFixed(3) : "—",
      note: "Isolation Forest, zero labels used",
    },
    {
      label: "Recall (anomalous)",
      value: meta.model.recall_anomalous !== null && meta.model.recall_anomalous !== undefined
        ? meta.model.recall_anomalous.toFixed(3) : "—",
      note: `Precision ${meta.model.precision_anomalous !== null && meta.model.precision_anomalous !== undefined
        ? meta.model.precision_anomalous.toFixed(3) : "—"}`,
    },
  ];
  grid.innerHTML = tiles
    .map(
      (t) => `
    <div class="metric-tile">
      <div class="label">${t.label}</div>
      <div class="value">${t.value}</div>
      <div class="note">${t.note}</div>
    </div>`
    )
    .join("");

  const impList = document.getElementById("importance-list");
  const maxImp = Math.max(...meta.top_features.map((f) => f.importance), 0.0001);
  impList.innerHTML = meta.top_features
    .map(
      (f) => `
    <div class="imp-row">
      <div class="name">${f.feature}</div>
      <div class="imp-bar-track"><div class="imp-bar-fill" style="width:${((f.importance / maxImp) * 100).toFixed(1)}%"></div></div>
      <div class="pct">${(f.importance * 100).toFixed(1)}%</div>
    </div>`
    )
    .join("");
}

// ---------------------------------------------------------------
// Segment list (left rail)
// ---------------------------------------------------------------
async function loadSegments() {
  const res = await fetch("/api/segments");
  const segments = await res.json();
  state.segments = segments;

  const list = document.getElementById("segment-list");
  list.innerHTML = segments
    .map(
      (s) => `
    <div class="segment-item" data-seg="${s.segment_id}">
      <div class="row1">
        <span class="seg-id">SEG-${String(s.segment_id).padStart(3, "0")}</span>
        <span class="badge ${alertClass(s.last_alert_level)}">${s.last_alert_level}</span>
      </div>
      <div class="incidents">${s.num_readings} readings · ${s.num_incidents} historical incidents</div>
    </div>`
    )
    .join("");

  list.querySelectorAll(".segment-item").forEach((el) => {
    el.addEventListener("click", () => selectSegment(parseInt(el.dataset.seg, 10)));
  });

  const wiSelect = document.getElementById("wi-segment");
  wiSelect.innerHTML = segments
    .map((s) => `<option value="${s.segment_id}">SEG-${String(s.segment_id).padStart(3, "0")}</option>`)
    .join("");

  if (segments.length) selectSegment(segments[0].segment_id);
}

async function selectSegment(segId) {
  stopPlayback();
  state.selectedSegment = segId;

  document.querySelectorAll(".segment-item").forEach((el) => {
    el.classList.toggle("selected", parseInt(el.dataset.seg, 10) === segId);
  });

  const res = await fetch(`/api/simulate/${segId}`);
  const data = await res.json();
  state.simData = data;
  state.tickIndex = data.length - 1;

  ensureSchematic();
  buildChart();

  const scrub = document.getElementById("scrub");
  scrub.max = data.length - 1;
  scrub.value = state.tickIndex;

  renderTick(state.tickIndex);
}

// ---------------------------------------------------------------
// Pipeline schematic (built once, then updated in place per tick)
// ---------------------------------------------------------------
function ensureSchematic() {
  const holder = document.getElementById("schematic-holder");
  if (holder.dataset.built === "1") return;
  holder.innerHTML = buildSchematicSVG();
  holder.dataset.built = "1";
}

function buildSchematicSVG() {
  return `
  <svg viewBox="0 0 640 130" width="100%" height="130" preserveAspectRatio="xMidYMid meet">
    <line x1="40" y1="65" x2="600" y2="65" stroke="#2a3341" stroke-width="10" stroke-linecap="round"/>
    <line id="sch-flow" class="sch-flow-dash" x1="40" y1="65" x2="600" y2="65" stroke="#4fb8e0" stroke-width="4" stroke-linecap="round"/>

    <circle cx="40" cy="65" r="10" fill="#171d26" stroke="#4d5665" stroke-width="2"/>
    <text x="40" y="95" class="sch-label">INLET</text>

    <g transform="translate(190,65)">
      <rect id="sch-valve" class="sch-glow" x="-14" y="-14" width="28" height="28" rx="4" fill="#4d5665"/>
      <text y="32" class="sch-label">VALVE</text>
      <text id="sch-valve-label" y="44" class="sch-label" style="fill:#e7ecf2;">—</text>
    </g>

    <g transform="translate(340,65)">
      <circle r="18" fill="#171d26" stroke="#2a3341" stroke-width="2"/>
      <g id="sch-pump" class="sch-pump-spinner">
        <path d="M0,-12 L4,-2 L14,0 L4,2 L0,12 L-4,2 L-14,0 L-4,-2 Z" fill="#4d5665"/>
      </g>
      <text y="32" class="sch-label">PUMP</text>
      <text id="sch-pump-label" y="44" class="sch-label" style="fill:#e7ecf2;">—</text>
    </g>

    <g transform="translate(480,65)">
      <rect id="sch-compressor" class="sch-glow" x="-16" y="-14" width="32" height="28" rx="4" fill="#4d5665"/>
      <text y="32" class="sch-label">COMPRESSOR</text>
      <text id="sch-compressor-label" y="44" class="sch-label" style="fill:#e7ecf2;">—</text>
    </g>

    <circle cx="600" cy="65" r="10" fill="#171d26" stroke="#4d5665" stroke-width="2"/>
    <text x="600" y="95" class="sch-label">OUTLET</text>
  </svg>`;
}

function updateSchematic(tick) {
  const r = tick.raw;
  const valve = document.getElementById("sch-valve");
  const valveLabel = document.getElementById("sch-valve-label");
  const pump = document.getElementById("sch-pump");
  const pumpPath = pump.querySelector("path");
  const pumpLabel = document.getElementById("sch-pump-label");
  const compressor = document.getElementById("sch-compressor");
  const compLabel = document.getElementById("sch-compressor-label");
  const flow = document.getElementById("sch-flow");

  valve.setAttribute("fill", VALVE_COLORS[r.valve_status] ?? "#4d5665");
  valveLabel.textContent = VALVE_NAMES[r.valve_status] ?? "—";

  pump.classList.toggle("spinning", r.pump_state === 1);
  pumpPath.setAttribute("fill", r.pump_state === 1 ? "#4fb8e0" : "#4d5665");
  pumpLabel.textContent = r.pump_state === 1 ? "RUNNING" : "OFF";

  compressor.setAttribute("fill", r.compressor_state === 1 ? "#4fb8e0" : "#4d5665");
  compLabel.textContent = r.compressor_state === 1 ? "RUNNING" : "OFF";

  const flowing = r.valve_status !== 0 && r.flow_rate > 0.5;
  flow.classList.toggle("paused", !flowing);
}

// ---------------------------------------------------------------
// Gauge + readouts + factors
// ---------------------------------------------------------------
function updateGauge(riskPct, alertLevel) {
  const arc = document.getElementById("gauge-arc");
  const offset = ARC_LENGTH * (1 - riskPct / 100);
  arc.setAttribute("stroke-dashoffset", offset.toFixed(2));
  const color =
    alertLevel === "Critical Leak Threat" ? "#e2483d" : alertLevel === "Warning" ? "#e8a33d" : "#45b880";
  arc.setAttribute("stroke", color);

  document.getElementById("risk-value").textContent = riskPct.toFixed(1);
  const pill = document.getElementById("alert-pill");
  pill.textContent = alertLevel;
  pill.className = "alert-pill " + alertClass(alertLevel);
}

function updateReadouts(tick) {
  const r = tick.raw;
  const tiles = [
    { label: "Pressure", value: `${fmt(r.pressure, 1)} bar` },
    { label: "Flow Rate", value: `${fmt(r.flow_rate, 2)} m³/s` },
    { label: "Temperature", value: `${fmt(r.temperature, 1)} °C` },
    { label: "Energy", value: `${fmt(r.energy_consumption, 1)} kW` },
    { label: "Pump Speed", value: `${fmt(r.pump_speed, 0)} RPM` },
    { label: "Valve", value: VALVE_NAMES[r.valve_status] ?? "—" },
    { label: "Compressor", value: r.compressor_state === 1 ? "Running" : "Off" },
    { label: "Threshold Alarm", value: r.alarm_triggered === 1 ? "TRIGGERED" : "Clear" },
  ];
  document.getElementById("readout-grid").innerHTML = tiles
    .map(
      (t) => `
    <div class="readout-tile">
      <div class="label">${t.label}</div>
      <div class="value">${t.value}</div>
    </div>`
    )
    .join("");
}

function factorValue(f) {
  return f.shap_contribution !== undefined ? f.shap_contribution : f.model_importance;
}

function updateFactors(tick) {
  const wrap = document.getElementById("factors-list");
  const factors = tick.top_contributing_factors || [];
  if (!factors.length) {
    wrap.innerHTML = `<p style="color:var(--text-dim); font-size:12px;">No attribution available.</p>`;
    return;
  }
  const maxAbs = Math.max(...factors.map((f) => Math.abs(factorValue(f))), 0.001);
  wrap.innerHTML = factors
    .map((f) => {
      const v = factorValue(f);
      const pct = Math.min(100, (Math.abs(v) / maxAbs) * 50);
      const cls = v >= 0 ? "pos" : "neg";
      return `
      <div class="factor-row">
        <div>
          <div class="factor-name">${f.feature}</div>
          <span class="factor-reading">reading: ${f.current_reading}</span>
          <div class="factor-bar-track"><div class="factor-bar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
        </div>
        <div class="factor-val">${v >= 0 ? "+" : ""}${v.toFixed(2)}</div>
      </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------
// Tick rendering (drives schematic + gauge + readouts + factors + chart marker)
// ---------------------------------------------------------------
function renderTick(index) {
  const tick = state.simData[index];
  if (!tick) return;
  state.tickIndex = index;

  document.getElementById("schematic-segid").textContent = tick.segment_id;
  document.getElementById("schematic-ts").textContent = tick.timestamp;

  updateSchematic(tick);
  updateGauge(tick.risk_score_pct, tick.alert_level);
  updateReadouts(tick);
  updateFactors(tick);

  document.getElementById("scrub").value = index;
  document.getElementById("scrub-label").textContent = `reading ${index + 1} / ${state.simData.length}`;

  updateChartMarker(index);
}

// ---------------------------------------------------------------
// Risk timeline chart
// ---------------------------------------------------------------
function buildChart() {
  const ctx = document.getElementById("risk-chart").getContext("2d");
  const labels = state.simData.map((d, i) => i + 1);
  const riskData = state.simData.map((d) => d.risk_score_pct);
  const pointColors = state.simData.map((d) =>
    d.alert_level === "Critical Leak Threat" ? "#e2483d" : d.alert_level === "Warning" ? "#e8a33d" : "#45b880"
  );

  if (state.chart) state.chart.destroy();
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Risk Score %",
          data: riskData,
          borderColor: "#4fb8e0",
          backgroundColor: "rgba(79,184,224,0.08)",
          pointBackgroundColor: pointColors,
          pointRadius: 3,
          tension: 0.25,
          fill: true,
        },
        {
          label: "Now",
          data: state.simData.map((d, i) => (i === state.tickIndex ? d.risk_score_pct : null)),
          borderColor: "transparent",
          pointBackgroundColor: "#ffffff",
          pointBorderColor: "#4fb8e0",
          pointBorderWidth: 2,
          pointRadius: 6,
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { ticks: { color: "#7c8798", font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#2a3341" } },
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#7c8798", font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "#2a3341" },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function updateChartMarker(index) {
  if (!state.chart) return;
  state.chart.data.datasets[1].data = state.simData.map((d, i) => (i === index ? d.risk_score_pct : null));
  state.chart.update("none");
}

// ---------------------------------------------------------------
// Playback controls
// ---------------------------------------------------------------
function setupPlaybackControls() {
  document.getElementById("play-btn").addEventListener("click", togglePlayback);
  document.getElementById("scrub").addEventListener("input", (e) => {
    stopPlayback();
    renderTick(parseInt(e.target.value, 10));
  });
  document.getElementById("speed-select").addEventListener("change", () => {
    if (state.playing) {
      stopPlayback();
      togglePlayback();
    }
  });
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (!state.simData.length) return;
  state.playing = true;
  document.getElementById("play-btn").textContent = "❚❚";
  const speed = parseInt(document.getElementById("speed-select").value, 10);
  state.playTimer = setInterval(() => {
    let next = state.tickIndex + 1;
    if (next >= state.simData.length) next = 0;
    renderTick(next);
  }, speed);
}

function stopPlayback() {
  state.playing = false;
  const btn = document.getElementById("play-btn");
  if (btn) btn.textContent = "▶";
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

// ---------------------------------------------------------------
// What-If Simulator
// ---------------------------------------------------------------
function setupWhatIfForm() {
  const sliders = [
    ["wi-pressure", "wi-pressure-val", 1],
    ["wi-flow", "wi-flow-val", 1],
    ["wi-temp", "wi-temp-val", 1],
    ["wi-pumpspeed", "wi-pumpspeed-val", 0],
    ["wi-energy", "wi-energy-val", 1],
  ];
  sliders.forEach(([inputId, labelId, digits]) => {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    input.addEventListener("input", () => {
      label.textContent = parseFloat(input.value).toFixed(digits);
    });
  });

  document.getElementById("wi-use-history").addEventListener("change", (e) => {
    document.getElementById("wi-segment-field").style.display = e.target.checked ? "block" : "none";
  });

  document.getElementById("wi-predict-btn").addEventListener("click", runWhatIf);
}

async function runWhatIf() {
  const useHistory = document.getElementById("wi-use-history").checked;
  const payload = {
    pressure: parseFloat(document.getElementById("wi-pressure").value),
    flow_rate: parseFloat(document.getElementById("wi-flow").value),
    temperature: parseFloat(document.getElementById("wi-temp").value),
    valve_status: parseInt(document.getElementById("wi-valve").value, 10),
    pump_state: parseInt(document.getElementById("wi-pump-state").value, 10),
    pump_speed: parseFloat(document.getElementById("wi-pumpspeed").value),
    compressor_state: parseInt(document.getElementById("wi-compressor-state").value, 10),
    energy_consumption: parseFloat(document.getElementById("wi-energy").value),
    alarm_triggered: parseInt(document.getElementById("wi-alarm").value, 10),
  };
  if (useHistory) {
    payload.use_history = true;
    payload.segment_id = parseInt(document.getElementById("wi-segment").value, 10);
  }

  const resultEl = document.getElementById("wi-result");
  resultEl.textContent = "Scoring...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) {
      resultEl.innerHTML = `<div style="color:var(--critical);">Error: ${data.error}</div>`;
      return;
    }
    renderWhatIfResult(data);
  } catch (err) {
    resultEl.innerHTML = `<div style="color:var(--critical);">Request failed: ${err}</div>`;
  }
}

function renderWhatIfResult(data) {
  const resultEl = document.getElementById("wi-result");
  const cls = alertClass(data.alert_level);
  const factorsHtml = data.top_contributing_factors
    .map((f) => {
      const v = factorValue(f);
      return `<div class="factor-row" style="max-width:340px; margin:0 auto;">
        <div>
          <div class="factor-name">${f.feature}</div>
          <span class="factor-reading">reading: ${f.current_reading}</span>
        </div>
        <div class="factor-val">${v >= 0 ? "+" : ""}${v.toFixed(2)}</div>
      </div>`;
    })
    .join("");

  resultEl.innerHTML = `
    <div style="font-family:var(--font-mono); font-size:48px; color:var(--text);">${data.risk_score_pct.toFixed(
      1
    )}<span style="font-size:20px;color:var(--text-muted);">%</span></div>
    <div class="alert-pill ${cls}" style="font-size:13px;">${data.alert_level}</div>
    <div style="font-size:11.5px; color:var(--text-dim); margin-top:6px;">
      classifier probability ${(data.classifier_probability * 100).toFixed(1)}% &nbsp;·&nbsp;
      anomaly score ${(data.anomaly_score * 100).toFixed(1)}%
    </div>
    <div style="width:100%; margin-top:18px; text-align:left;">
      <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.07em; color:var(--text-muted); margin-bottom:6px;">
        Top contributing factors
      </div>
      ${factorsHtml}
    </div>`;
  resultEl.style.color = "var(--text)";
}

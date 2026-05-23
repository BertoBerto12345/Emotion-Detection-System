/* ============================================================
   Student Emotion Detection System – Frontend Logic
   Browser webcam → canvas → base64 → Flask /analyze API
   ============================================================ */

// ── Emotion metadata ──────────────────────────────────────────
const EMOTION_META = {
  Happy:     { emoji: "😄", color: "#22c55e" },
  Sad:       { emoji: "😢", color: "#ef4444" },
  Angry:     { emoji: "😠", color: "#f97316" },
  Confused:  { emoji: "😕", color: "#06b6d4" },
  Bored:     { emoji: "😑", color: "#a855f7" },
  Surprised: { emoji: "😲", color: "#eab308" },
  Neutral:   { emoji: "😐", color: "#94a3b8" },
  "No Face": { emoji: "😶", color: "#64748b" },
  "Waiting…":{ emoji: "⏳", color: "#64748b" },
  "Error":   { emoji: "⚠️", color: "#ef4444" },
};

function meta(emotion) {
  return EMOTION_META[emotion] || { emoji: "❓", color: "#94a3b8" };
}

// ── DOM refs ──────────────────────────────────────────────────
const video         = document.getElementById("video");
const canvas        = document.getElementById("canvas");
const ctx           = canvas.getContext("2d");
const emojiEl       = document.getElementById("emotion-emoji");
const labelEl       = document.getElementById("emotion-label");
const confBar       = document.getElementById("confidence-bar");
const confText      = document.getElementById("confidence-text");
const scoreGrid     = document.getElementById("score-grid");
const summaryBars   = document.getElementById("summary-bars");
const totalBadge    = document.getElementById("total-detections");
const logList       = document.getElementById("log-list");
const statusBadge   = document.getElementById("status-badge");
const camStatus     = document.getElementById("cam-status");
const camOverlay    = document.getElementById("cam-overlay");
const btnStart      = document.getElementById("btn-start");
const btnStop       = document.getElementById("btn-stop");

// ── State ─────────────────────────────────────────────────────
let stream          = null;
let analyzeInterval = null;
let summaryInterval = null;
let logInterval     = null;
let lastEmotion     = "";
let isAnalyzing     = false;

// ── Camera ────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    camOverlay.style.display = "none";

    btnStart.disabled = true;
    btnStop.disabled  = false;
    camStatus.textContent = "Active";
    camStatus.className   = "badge badge-live";
    statusBadge.textContent = "● LIVE";
    statusBadge.className   = "badge badge-live";

    startPolling();
  } catch (err) {
    camOverlay.style.display = "flex";
    camOverlay.querySelector("span").textContent =
      "⚠️ Camera access denied. Please allow camera permission.";
    console.error("Camera error:", err);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  camOverlay.style.display = "flex";
  camOverlay.querySelector("span").textContent = "📷 Click Start to enable camera";

  btnStart.disabled = false;
  btnStop.disabled  = true;
  camStatus.textContent = "Stopped";
  camStatus.className   = "badge badge-offline";
  statusBadge.textContent = "● OFFLINE";
  statusBadge.className   = "badge badge-offline";

  stopPolling();
}

// ── Capture & analyze ─────────────────────────────────────────
async function captureAndAnalyze() {
  if (isAnalyzing || !stream) return;
  isAnalyzing = true;

  try {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL("image/jpeg", 0.7);

    const res  = await fetch("/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: dataURL }),
    });
    const data = await res.json();
    updateEmotionPanel(data);
  } catch (e) {
    console.warn("Analyze error:", e);
  } finally {
    isAnalyzing = false;
  }
}

// ── Update UI ─────────────────────────────────────────────────
function updateEmotionPanel(data) {
  const emotion    = data.emotion    || "No Face";
  const confidence = data.confidence || 0;
  const scores     = data.all_scores || {};
  const m          = meta(emotion);

  if (emotion !== lastEmotion) {
    emojiEl.textContent  = m.emoji;
    emojiEl.style.transform = "scale(1.3)";
    setTimeout(() => { emojiEl.style.transform = "scale(1)"; }, 250);
    lastEmotion = emotion;
  }

  labelEl.textContent = emotion;
  labelEl.style.color = m.color;

  const pct = Math.min(confidence, 100);
  confBar.style.width      = pct + "%";
  confBar.style.background = `linear-gradient(90deg, ${m.color}99, ${m.color})`;
  confText.textContent     = pct.toFixed(1) + "%";

  if (Object.keys(scores).length > 0) {
    scoreGrid.innerHTML = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .map(([emo, val]) => {
        const sm = meta(emo);
        return `
          <div class="score-item">
            <div class="score-item-label">${sm.emoji} ${emo}</div>
            <div class="score-item-bar-track">
              <div class="score-item-bar" style="width:${Math.min(val,100)}%;background:${sm.color}"></div>
            </div>
            <div class="score-item-value">${val.toFixed(1)}%</div>
          </div>`;
      }).join("");
  }
}

function updateSummary(data) {
  const summary = data.summary || {};
  const total   = data.total   || 0;
  totalBadge.textContent = `${total} detections`;

  if (Object.keys(summary).length === 0) {
    summaryBars.innerHTML = `<p style="color:var(--text-muted);font-size:.8rem;padding:.5rem 0">No data yet…</p>`;
    return;
  }

  summaryBars.innerHTML = Object.entries(summary)
    .sort((a, b) => b[1].count - a[1].count)
    .map(([emo, info]) => {
      const m = meta(emo);
      return `
        <div class="summary-row">
          <span class="summary-label">${m.emoji} ${emo}</span>
          <div class="summary-track">
            <div class="summary-fill" style="width:${info.percent}%;background:${m.color}"></div>
          </div>
          <span class="summary-pct">${info.percent}%</span>
        </div>`;
    }).join("");
}

function updateLog(entries) {
  if (!entries || entries.length === 0) {
    logList.innerHTML = `<p style="color:var(--text-muted);font-size:.8rem;padding:.5rem 0">No entries yet…</p>`;
    return;
  }
  const recent = [...entries].reverse().slice(0, 30);
  logList.innerHTML = recent.map(entry => {
    const m  = meta(entry.emotion);
    const ts = new Date(entry.timestamp).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
    return `
      <div class="log-entry">
        <div class="log-dot" style="background:${m.color}"></div>
        <span class="log-emotion" style="color:${m.color}">${entry.emotion}</span>
        <span class="log-conf">${(entry.confidence||0).toFixed(1)}%</span>
        <span class="log-time">${ts}</span>
      </div>`;
  }).join("");
}

// ── Polling ───────────────────────────────────────────────────
function startPolling() {
  analyzeInterval = setInterval(captureAndAnalyze, 1000);
  summaryInterval = setInterval(fetchSummary, 3000);
  logInterval     = setInterval(fetchLog,     2000);
}

function stopPolling() {
  clearInterval(analyzeInterval); analyzeInterval = null;
  clearInterval(summaryInterval); summaryInterval = null;
  clearInterval(logInterval);     logInterval     = null;
}

async function fetchSummary() {
  try {
    const res  = await fetch("/emotion_summary");
    const data = await res.json();
    updateSummary(data);
  } catch (e) { console.warn("Summary fetch failed:", e); }
}

async function fetchLog() {
  try {
    const res  = await fetch("/emotion_log");
    const data = await res.json();
    updateLog(data);
  } catch (e) { console.warn("Log fetch failed:", e); }
}

// ── Export / Clear ────────────────────────────────────────────
function exportLog() {
  window.location.href = "/export_log";
}

async function clearLog() {
  if (!confirm("Clear all emotion log data?")) return;
  try {
    await fetch("/clear_log", { method: "POST" });
    updateLog([]);
    updateSummary({ summary: {}, total: 0 });
  } catch (e) { alert("Failed to clear log."); }
}

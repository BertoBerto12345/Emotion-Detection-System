/* ============================================================
   Student Emotion Detection System – Frontend Logic
   ============================================================ */

// ── Emotion metadata ──────────────────────────────────────────
const EMOTION_META = {
  Happy:     { emoji: "😄", color: "#22c55e", cssClass: "color-happy"     },
  Sad:       { emoji: "😢", color: "#ef4444", cssClass: "color-sad"       },
  Angry:     { emoji: "😠", color: "#f97316", cssClass: "color-angry"     },
  Confused:  { emoji: "😕", color: "#06b6d4", cssClass: "color-confused"  },
  Bored:     { emoji: "😑", color: "#a855f7", cssClass: "color-bored"     },
  Surprised: { emoji: "😲", color: "#eab308", cssClass: "color-surprised" },
  Neutral:   { emoji: "😐", color: "#94a3b8", cssClass: "color-neutral"   },
  "No Face": { emoji: "😶", color: "#64748b", cssClass: "color-neutral"   },
  "Initializing...": { emoji: "⏳", color: "#64748b", cssClass: "color-neutral" },
};

// ── DOM refs ──────────────────────────────────────────────────
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
const videoFeed     = document.getElementById("video-feed");
const noFaceOverlay = document.getElementById("no-face-overlay");

// ── State ─────────────────────────────────────────────────────
let pollingInterval  = null;
let summaryInterval  = null;
let logInterval      = null;
let cameraActive     = true;
let lastEmotion      = "";

// ── Helpers ───────────────────────────────────────────────────
function meta(emotion) {
  return EMOTION_META[emotion] || { emoji: "❓", color: "#94a3b8", cssClass: "color-neutral" };
}

function formatTime(isoString) {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch { return ""; }
}

// ── Update current emotion panel ─────────────────────────────
function updateEmotionPanel(data) {
  const emotion    = data.emotion    || "Initializing...";
  const confidence = data.confidence || 0;
  const scores     = data.all_scores || {};
  const m          = meta(emotion);

  // Emoji + label
  if (emotion !== lastEmotion) {
    emojiEl.textContent = m.emoji;
    emojiEl.style.transform = "scale(1.3)";
    setTimeout(() => { emojiEl.style.transform = "scale(1)"; }, 250);
    lastEmotion = emotion;
  }

  labelEl.textContent = emotion;
  labelEl.style.color = m.color;

  // Confidence bar
  const pct = Math.min(confidence, 100);
  confBar.style.width  = pct + "%";
  confBar.style.background = `linear-gradient(90deg, ${m.color}99, ${m.color})`;
  confText.textContent = pct.toFixed(1) + "%";

  // No-face overlay
  noFaceOverlay.style.display = (emotion === "No Face") ? "flex" : "none";

  // Score breakdown grid
  if (Object.keys(scores).length > 0) {
    scoreGrid.innerHTML = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .map(([emo, val]) => {
        const sm = meta(emo);
        return `
          <div class="score-item">
            <div class="score-item-label">${sm.emoji} ${emo}</div>
            <div class="score-item-bar-track">
              <div class="score-item-bar" style="width:${Math.min(val,100)}%; background:${sm.color}"></div>
            </div>
            <div class="score-item-value">${val.toFixed(1)}%</div>
          </div>`;
      }).join("");
  }
}

// ── Update session summary ────────────────────────────────────
function updateSummary(data) {
  const summary = data.summary || {};
  const total   = data.total   || 0;

  totalBadge.textContent = `${total} detections`;

  if (Object.keys(summary).length === 0) {
    summaryBars.innerHTML = `<p style="color:var(--text-muted);font-size:.8rem;padding:.5rem 0">No data yet…</p>`;
    return;
  }

  const sorted = Object.entries(summary).sort((a, b) => b[1].count - a[1].count);

  summaryBars.innerHTML = sorted.map(([emo, info]) => {
    const m = meta(emo);
    return `
      <div class="summary-row">
        <span class="summary-label">${m.emoji} ${emo}</span>
        <div class="summary-track">
          <div class="summary-fill" style="width:${info.percent}%; background:${m.color}"></div>
        </div>
        <span class="summary-pct">${info.percent}%</span>
      </div>`;
  }).join("");
}

// ── Update recent log ─────────────────────────────────────────
function updateLog(entries) {
  if (!entries || entries.length === 0) {
    logList.innerHTML = `<p style="color:var(--text-muted);font-size:.8rem;padding:.5rem 0">No entries yet…</p>`;
    return;
  }

  // Show last 30, newest first
  const recent = [...entries].reverse().slice(0, 30);

  logList.innerHTML = recent.map(entry => {
    const m = meta(entry.emotion);
    return `
      <div class="log-entry">
        <div class="log-dot" style="background:${m.color}"></div>
        <span class="log-emotion" style="color:${m.color}">${entry.emotion}</span>
        <span class="log-conf">${(entry.confidence || 0).toFixed(1)}%</span>
        <span class="log-time">${formatTime(entry.timestamp)}</span>
      </div>`;
  }).join("");
}

// ── Fetch helpers ─────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function pollEmotion() {
  try {
    const data = await fetchJSON("/emotion_status");
    updateEmotionPanel(data);
  } catch (e) {
    console.warn("Emotion poll failed:", e);
  }
}

async function pollSummary() {
  try {
    const data = await fetchJSON("/emotion_summary");
    updateSummary(data);
  } catch (e) {
    console.warn("Summary poll failed:", e);
  }
}

async function pollLog() {
  try {
    const data = await fetchJSON("/emotion_log");
    updateLog(data);
  } catch (e) {
    console.warn("Log poll failed:", e);
  }
}

// ── Camera controls ───────────────────────────────────────────
function startCamera() {
  if (cameraActive) return;
  videoFeed.src = "/video_feed?" + Date.now(); // force reload
  cameraActive = true;
  camStatus.textContent   = "Active";
  camStatus.className     = "badge badge-live";
  statusBadge.textContent = "● LIVE";
  statusBadge.className   = "badge badge-live";
  startPolling();
}

async function stopCamera() {
  try {
    await fetch("/stop_camera", { method: "POST" });
  } catch (e) { /* ignore */ }
  videoFeed.src = "";
  cameraActive = false;
  camStatus.textContent   = "Stopped";
  camStatus.className     = "badge badge-offline";
  statusBadge.textContent = "● OFFLINE";
  statusBadge.className   = "badge badge-offline";
  stopPolling();
}

// ── Export / clear ────────────────────────────────────────────
function exportLog() {
  window.location.href = "/export_log";
}

async function clearLog() {
  if (!confirm("Clear all emotion log data for this session?")) return;
  try {
    await fetch("/clear_log", { method: "POST" });
    updateLog([]);
    updateSummary({ summary: {}, total: 0 });
    totalBadge.textContent = "0 detections";
  } catch (e) {
    alert("Failed to clear log.");
  }
}

// ── Polling lifecycle ─────────────────────────────────────────
function startPolling() {
  if (pollingInterval) return;
  pollingInterval = setInterval(pollEmotion,  800);
  summaryInterval = setInterval(pollSummary, 3000);
  logInterval     = setInterval(pollLog,     2000);
  // Immediate first fetch
  pollEmotion();
  pollSummary();
  pollLog();
}

function stopPolling() {
  clearInterval(pollingInterval); pollingInterval = null;
  clearInterval(summaryInterval); summaryInterval = null;
  clearInterval(logInterval);     logInterval     = null;
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  startPolling();

  // Handle video feed errors (e.g., camera not available)
  videoFeed.addEventListener("error", () => {
    noFaceOverlay.style.display = "flex";
    noFaceOverlay.querySelector("span").textContent = "📷 Camera unavailable";
  });
});

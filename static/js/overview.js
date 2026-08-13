document.addEventListener("DOMContentLoaded", async () => {
  const { api, percent } = window.MindFuse;
  try {
    const [health, metrics] = await Promise.all([api("/api/health"), api("/api/metrics")]);
    const summary = metrics.summary?.metrics || {};
    const values = {
      face: percent(summary.face_macro_f1),
      audio: percent(summary.audio_macro_f1),
      numerical: percent(summary.numerical_macro_f1),
      regression: Number.isFinite(Number(summary.regression_mean_rmse)) ? Number(summary.regression_mean_rmse).toFixed(2) : "Unavailable"
    };
    Object.entries(values).forEach(([name, value]) => {
      const node = document.querySelector(`[data-metric="${name}"]`);
      if (node) node.textContent = value;
    });
    const models = health.model_status.models || {};
    ["face", "audio", "numerical"].forEach(name => {
      const node = document.getElementById(`${name}-model-state`);
      if (models[name]) node?.classList.add("ready");
    });
    const dataset = health.dataset || {};
    const set = (id, value) => { const node = document.getElementById(id); if (node) node.textContent = value == null ? "Unavailable" : Number(value).toLocaleString(); };
    set("face-count", dataset.face_files);
    set("audio-count", dataset.audio_files);
    set("numerical-count", dataset.numerical_rows);
  } catch (error) {
    window.MindFuse.toast(error.message, "error");
  }
});


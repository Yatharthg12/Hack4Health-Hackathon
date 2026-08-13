document.addEventListener("DOMContentLoaded", async () => {
  const { api, escapeHtml, label, fixed } = window.MindFuse;
  try {
    const metrics = await api("/api/metrics");
    const items = metrics.numerical_classification?.global_feature_importance || [];
    const max = Math.max(...items.map(item => Math.max(0, item.importance)), 1e-9);
    const container = document.getElementById("importance-bars");
    container.classList.remove("loading-block");
    container.innerHTML = items.slice(0, 14).map(item => `<div class="importance-row"><span>${escapeHtml(label(item.feature))}</span><i><b style="width:${Math.max(0, item.importance) / max * 100}%"></b></i><strong>${fixed(item.importance, 4)}</strong></div>`).join("") || "Measured importance is unavailable until training completes.";
    const weights = metrics.fusion_weights?.weights || {};
    document.getElementById("fusion-weight-list").innerHTML = Object.entries(weights).map(([name, value]) => `<span class="chip">${escapeHtml(label(name))} <strong>${fixed(value, 3)}</strong></span>`).join("") || `<span class="chip">Validation weights unavailable until training</span>`;
  } catch (error) { window.MindFuse.toast(error.message, "error"); }
});


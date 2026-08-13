(() => {
  const root = document.documentElement;
  const savedTheme = localStorage.getItem("mindfuse-theme");
  if (savedTheme) root.dataset.theme = savedTheme;

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
  const label = value => String(value ?? "").replaceAll("_", " ");
  const percent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "Unavailable";
  const fixed = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "Unavailable";

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const type = response.headers.get("content-type") || "";
    const payload = type.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      throw new Error(payload?.error?.message || payload?.message || payload || `Request failed (${response.status})`);
    }
    return payload;
  }

  function toast(message, kind = "info") {
    const region = document.getElementById("toast-region");
    if (!region) return;
    const item = document.createElement("div");
    item.className = `toast ${kind === "error" ? "error" : ""}`;
    item.textContent = message;
    region.append(item);
    setTimeout(() => item.remove(), 4500);
  }

  function probabilityBars(probabilities, compact = false) {
    const sorted = Object.entries(probabilities || {}).sort((first, second) => second[1] - first[1]);
    const className = compact ? "mini-bars" : "probability-list";
    const rowName = compact ? "mini-bar" : "probability-row";
    return `<div class="${className}">${sorted.map(([name, value]) =>
      `<div class="${rowName}"><span>${escapeHtml(label(name))}</span><i><b style="width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></b></i><strong>${percent(value)}</strong></div>`
    ).join("")}</div>`;
  }

  window.MindFuse = { api, toast, escapeHtml, label, percent, fixed, probabilityBars };

  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("mindfuse-theme", root.dataset.theme);
  });
  document.getElementById("mobile-menu")?.addEventListener("click", () => document.body.classList.toggle("nav-open"));
  document.querySelectorAll(".nav-item").forEach(item => item.addEventListener("click", () => document.body.classList.remove("nav-open")));

  api("/api/health").then(health => {
    const dot = document.getElementById("sidebar-status");
    const text = document.getElementById("sidebar-status-text");
    const readyCount = Object.values(health.model_status.models || {}).filter(Boolean).length;
    const total = Object.keys(health.model_status.models || {}).length;
    dot?.classList.add(health.status === "ready" ? "ready" : "error");
    if (text) text.textContent = health.status === "ready" ? "All models ready" : `${readyCount}/${total} models ready`;
  }).catch(() => {
    document.getElementById("sidebar-status")?.classList.add("error");
    const text = document.getElementById("sidebar-status-text");
    if (text) text.textContent = "API unavailable";
  });
})();

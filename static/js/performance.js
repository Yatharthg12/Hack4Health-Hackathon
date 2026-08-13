document.addEventListener("DOMContentLoaded", async () => {
  const { api, escapeHtml, label, percent, fixed } = window.MindFuse;
  const metricCell = (name, value, asPercent = true) => `<div class="metric-cell"><span>${name}</span><strong>${asPercent ? percent(value) : fixed(value, 3)}</strong></div>`;
  const table = (headers, rows) => `<div class="table-wrap"><table class="data-table"><thead><tr>${headers.map(header => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  try {
    const metrics = await api("/api/metrics");
    const status = document.getElementById("performance-status");
    const configurations = [
      ["Facial emotion classifier", "face", metrics.face, "face_confusion_matrix.png", "face_training_history.png"],
      ["Speech emotion CNN", "audio", metrics.audio, "audio_confusion_matrix.png", "audio_training_history.png"],
      ["Numerical stress classifier", "numerical", metrics.numerical_classification, "numerical_confusion_matrix.png", "numerical_feature_importance.png"]
    ];
    const available = configurations.filter(entry => entry[2]?.test_metrics);
    status.textContent = available.length === 3 ? "All primary classification artifacts loaded from held-out evaluations." : `${available.length}/3 classification artifacts available. Train missing models to populate remaining results.`;
    const container = document.getElementById("classification-performance");
    container.innerHTML = configurations.map(([title, name, artifact, confusion, secondary]) => {
      const measured = artifact?.test_metrics;
      if (!measured) return `<article class="panel"><span class="kicker">${title}</span><h2>Unavailable</h2><p>Run the complete training workflow to generate real evaluation artifacts.</p></article>`;
      const perClass = Object.entries(measured.per_class || {}).map(([className, values]) => [label(className), fixed(values.precision), fixed(values.recall), fixed(values["f1-score"]), String(values.support)]);
      return `<article class="panel performance-card"><div class="panel-heading"><div><span class="kicker">Held-out evaluation · n=${measured.support}</span><h2>${escapeHtml(title)}</h2><small>${escapeHtml(artifact.split_method || artifact.model || "Validation-selected estimator")}</small></div><span class="tag">Real artifact</span></div><div class="metric-row">${metricCell("Accuracy", measured.accuracy)}${metricCell("Balanced accuracy", measured.balanced_accuracy)}${metricCell("Macro precision", measured.macro_precision)}${metricCell("Macro recall", measured.macro_recall)}${metricCell("Macro F1", measured.macro_f1)}${metricCell("ROC-AUC", measured.roc_auc_macro_ovr)}</div><div class="performance-visuals"><img src="/generated/${confusion}" alt="${title} confusion matrix"><img src="/generated/${secondary}" alt="${title} secondary evaluation plot"></div>${table(["Class", "Precision", "Recall", "F1", "Support"], perClass)}</article>`;
    }).join("");
    const regression = metrics.regression?.test_metrics;
    document.getElementById("regression-table").innerHTML = regression ? `${table(["Target", "MAE", "MSE", "RMSE", "R²", "Explained variance"], Object.entries(regression.per_target).map(([name, values]) => [label(name), fixed(values.mae), fixed(values.mse), fixed(values.rmse), fixed(values.r2), fixed(values.explained_variance)]))}<div class="performance-visuals"><img src="/generated/regression_predicted_vs_actual.png" alt="Regression predicted versus actual plots"></div>` : `<div class="notice">Regression metrics are unavailable until training completes.</div>`;
    function comparisons(target, title, items, field, lowerBetter = false) {
      const rows = Object.entries(items || {}); const values = rows.map(([, value]) => Number(value[field]));
      const high = Math.max(...values, 1e-9), low = Math.min(...values, 0);
      document.getElementById(target).innerHTML = `<h3>${title}</h3><div class="comparison-list">${rows.map(([name, value]) => { const score = Number(value[field]); const width = lowerBetter ? (high === low ? 100 : (high - score) / (high - low) * 85 + 15) : score / high * 100; return `<div class="comparison-item"><span>${escapeHtml(label(name))}</span><i><b style="width:${width}%"></b></i><strong>${fixed(score, 4)}</strong></div>`; }).join("") || "Unavailable"}</div>`;
    }
    comparisons("classifier-comparison", "Classification · validation macro-F1", metrics.numerical_classification?.model_selection, "macro_f1");
    comparisons("regressor-comparison", "Regression · normalized validation RMSE", metrics.regression?.model_selection, "mean_normalized_rmse", true);
  } catch (error) { document.getElementById("performance-status").classList.add("error"); document.getElementById("performance-status").textContent = error.message; }
});

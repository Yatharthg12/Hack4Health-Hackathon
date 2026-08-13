document.addEventListener("DOMContentLoaded", () => {
  const { toast } = window.MindFuse;
  const input = document.getElementById("batch-input");
  const zone = document.getElementById("batch-dropzone");
  const button = document.getElementById("run-batch");
  const status = document.getElementById("batch-status");
  function changed() {
    const file = input.files[0]; button.disabled = !file;
    document.getElementById("batch-meta").textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "Predictions, probabilities, and clamped D/A/S estimates will be appended.";
  }
  input.addEventListener("change", changed);
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", event => { if (event.dataTransfer.files.length) { const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]); input.files = transfer.files; changed(); } });
  button.addEventListener("click", async () => {
    const file = input.files[0]; if (!file) return;
    button.classList.add("is-loading"); button.disabled = true; status.hidden = false; status.textContent = "Validating schema and running models…";
    try {
      const data = new FormData(); data.append("csv", file);
      const response = await fetch("/api/predict/batch", { method: "POST", body: data });
      if (!response.ok) { const payload = await response.json(); throw new Error(payload.message || "Batch prediction failed"); }
      const blob = await response.blob(); const link = document.createElement("a");
      link.href = URL.createObjectURL(blob); link.download = "mindfuse_batch_results.csv"; link.click(); URL.revokeObjectURL(link.href);
      status.textContent = "Assessment complete. The processed CSV has been downloaded.";
    } catch (error) { status.classList.add("error"); status.textContent = error.message; toast(error.message, "error"); }
    finally { button.classList.remove("is-loading"); button.disabled = !input.files[0]; }
  });
});


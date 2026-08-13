document.addEventListener("DOMContentLoaded", () => {
  const { api, toast, escapeHtml, label, percent, probabilityBars } = window.MindFuse;
  const state = { face: null, audio: null, numerical: null, final: null };
  const revisions = { face: 0, audio: 0, numerical: 0 };
  const inflight = { face: null, audio: null, numerical: null };
  const faceInput = document.getElementById("face-input");
  const audioInput = document.getElementById("audio-input");
  const form = document.getElementById("numerical-form");
  const configPromise = api("/api/config");
  const healthPromise = api("/api/health");

  function setButtonLoading(button, loading) {
    button.classList.toggle("is-loading", loading);
    button.disabled = loading;
  }

  function setState(name, status, text) {
    const node = document.getElementById(`${name}-state`);
    node.textContent = text;
    node.className = `modality-state ${status}`;
  }

  function invalidate(name, status = "", text = "Optional") {
    revisions[name] += 1;
    state[name] = null;
    state.final = null;
    document.getElementById(`${name}-result`).hidden = true;
    document.getElementById("fusion-result").hidden = true;
    setState(name, status, text);
  }

  function configureDropzone(input, zone, onChange) {
    ["dragenter", "dragover"].forEach(event => zone.addEventListener(event, incident => {
      incident.preventDefault();
      zone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach(event => zone.addEventListener(event, incident => {
      incident.preventDefault();
      zone.classList.remove("dragover");
    }));
    zone.addEventListener("drop", incident => {
      if (incident.dataTransfer.files.length) {
        const transfer = new DataTransfer();
        transfer.items.add(incident.dataTransfer.files[0]);
        input.files = transfer.files;
        onChange();
      }
    });
    input.addEventListener("change", onChange);
  }

  configureDropzone(faceInput, document.getElementById("face-dropzone"), () => {
    const file = faceInput.files[0];
    invalidate("face", file ? "ready" : "", file ? "Selected" : "Optional");
    document.getElementById("analyze-face").disabled = !file;
    const preview = document.getElementById("face-preview");
    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
    } else {
      preview.hidden = true;
    }
  });

  configureDropzone(audioInput, document.getElementById("audio-dropzone"), () => {
    const file = audioInput.files[0];
    invalidate("audio", file ? "ready" : "", file ? "Selected" : "Optional");
    document.getElementById("analyze-audio").disabled = !file;
    document.getElementById("audio-error").hidden = true;
    document.getElementById("audio-file-meta").textContent = file
      ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`
      : "Mono/stereo WAV · speech content";
  });

  function renderModality(name, result) {
    const container = document.getElementById(`${name}-result`);
    const emotion = result.emotion
      ? `<div class="prediction-line"><span>Predicted emotion</span><strong>${escapeHtml(result.emotion)}</strong></div>`
      : "";
    const stress = `<div class="prediction-line"><span>Stress evidence</span><strong>${escapeHtml(label(result.stress_class))}</strong></div>`;
    const probabilities = result.emotion_probabilities || result.stress_probabilities;
    const image = result.face_explanation || result.gradcam || result.audio_explanation;
    const explanationNote = result.explanation?.available === false
      ? `<p class="profile-caption">${escapeHtml(result.explanation.message)}</p>`
      : "";
    container.innerHTML = `${emotion}${stress}${probabilityBars(probabilities, true)}${image ? `<img src="${image}" alt="Model explanation visualization">` : ""}${explanationNote}`;
    container.hidden = false;
  }

  async function analyzeFace() {
    if (inflight.face) return inflight.face;
    if (!faceInput.files[0]) return null;
    const button = document.getElementById("analyze-face");
    const revision = revisions.face;
    inflight.face = (async () => {
      setButtonLoading(button, true);
      setState("face", "ready", "Analyzing…");
      try {
        const data = new FormData();
        data.append("image", faceInput.files[0]);
        const result = await api("/api/predict/face", { method: "POST", body: data });
        if (revision !== revisions.face) return null;
        state.face = result;
        renderModality("face", result);
        setState("face", "ready", "Analyzed");
        return result;
      } catch (error) {
        if (revision === revisions.face) setState("face", "error", "Error");
        toast(error.message, "error");
        throw error;
      } finally {
        setButtonLoading(button, false);
        button.disabled = !faceInput.files[0];
        inflight.face = null;
      }
    })();
    return inflight.face;
  }

  async function analyzeAudio() {
    if (inflight.audio) return inflight.audio;
    if (!audioInput.files[0]) return null;
    const button = document.getElementById("analyze-audio");
    const errorNode = document.getElementById("audio-error");
    const revision = revisions.audio;
    inflight.audio = (async () => {
      setButtonLoading(button, true);
      setState("audio", "ready", "Analyzing…");
      errorNode.hidden = true;
      try {
        const data = new FormData();
        data.append("audio", audioInput.files[0]);
        const result = await api("/api/predict/audio", { method: "POST", body: data });
        if (revision !== revisions.audio) return null;
        state.audio = result;
        renderModality("audio", result);
        setState("audio", "ready", "Analyzed");
        return result;
      } catch (error) {
        if (revision === revisions.audio) {
          setState("audio", "error", "Error");
          errorNode.textContent = error.message;
          errorNode.hidden = false;
        }
        toast(error.message, "error");
        throw error;
      } finally {
        setButtonLoading(button, false);
        button.disabled = !audioInput.files[0];
        inflight.audio = null;
      }
    })();
    return inflight.audio;
  }

  function numericalPayload(requireValid = true) {
    if (requireValid && !form.reportValidity()) {
      throw new Error("Complete all 18 numerical fields within their accepted ranges.");
    }
    const payload = {};
    new FormData(form).forEach((value, key) => {
      if (value !== "") payload[key] = Number(value);
    });
    return payload;
  }

  async function analyzeNumerical() {
    if (inflight.numerical) return inflight.numerical;
    const button = document.getElementById("analyze-numerical");
    let payload;
    try {
      payload = numericalPayload(true);
    } catch (error) {
      toast(error.message, "error");
      throw error;
    }
    const revision = revisions.numerical;
    inflight.numerical = (async () => {
      setButtonLoading(button, true);
      setState("numerical", "ready", "Analyzing…");
      try {
        const result = await api("/api/predict/numerical", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (revision !== revisions.numerical) return null;
        state.numerical = result;
        renderModality("numerical", result);
        setState("numerical", "ready", "Analyzed");
        return result;
      } catch (error) {
        if (revision === revisions.numerical) setState("numerical", "error", "Error");
        toast(error.message, "error");
        throw error;
      } finally {
        setButtonLoading(button, false);
        inflight.numerical = null;
      }
    })();
    return inflight.numerical;
  }

  document.getElementById("analyze-face").addEventListener("click", () => analyzeFace().catch(() => {}));
  document.getElementById("analyze-audio").addEventListener("click", () => analyzeAudio().catch(() => {}));
  document.getElementById("analyze-numerical").addEventListener("click", () => analyzeNumerical().catch(() => {}));

  document.querySelectorAll("[data-profile]").forEach(button => button.addEventListener("click", async () => {
    try {
      const config = await configPromise;
      const profile = config.demo_profiles?.profiles?.[button.dataset.profile];
      if (!profile?.values) throw new Error("This demonstration profile is unavailable.");
      invalidate("numerical", "ready", `${profile.label} loaded`);
      Object.entries(profile.values).forEach(([name, value]) => {
        if (form.elements[name]) {
          form.elements[name].value = value;
          form.elements[name].setCustomValidity("");
        }
      });
      document.querySelectorAll("[data-profile]").forEach(item => item.classList.toggle("active", item === button));
      document.getElementById("profile-loaded").textContent = `${profile.label} loaded`;
      toast(`${profile.label} demonstration inputs loaded. Run analysis to obtain a genuine prediction.`);
    } catch (error) {
      toast(error.message, "error");
    }
  }));

  document.getElementById("clear-form").addEventListener("click", () => {
    form.reset();
    form.querySelectorAll("input").forEach(input => input.setCustomValidity(""));
    document.querySelectorAll("[data-profile]").forEach(item => item.classList.remove("active"));
    document.getElementById("profile-loaded").textContent = "";
    invalidate("numerical");
  });

  form.addEventListener("input", () => {
    document.querySelectorAll("[data-profile]").forEach(item => item.classList.remove("active"));
    document.getElementById("profile-loaded").textContent = "";
    invalidate("numerical", "ready", "Edited");
  });

  function reportProbabilities(probabilities) {
    return Object.entries(probabilities || {}).map(([name, value]) =>
      `<div class="report-probability"><span>${escapeHtml(label(name))}</span><i><b style="width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></b></i><strong>${percent(value)}</strong></div>`
    ).join("");
  }

  function reportModalityCard(name, evidence, contribution) {
    const modalityName = name === "face" ? "Facial expression" : name === "audio" ? "Speech acoustics" : "Behavior + physiology";
    const detail = evidence?.emotion
      ? `Predicted emotion: <strong>${escapeHtml(evidence.emotion)}</strong> (${percent(evidence.confidence)})`
      : `Predicted stress evidence: <strong>${escapeHtml(label(evidence?.stress_class))}</strong> (${percent(evidence?.confidence)})`;
    const technical = name === "face"
      ? `<div><span>Model</span><strong>${escapeHtml(evidence?.model || "Face classifier")}</strong></div><div><span>Explanation</span><strong>${escapeHtml(evidence?.explanation_method || "Model-tied sensitivity")}</strong></div>`
      : name === "audio"
        ? `<div><span>Duration</span><strong>${Number(evidence?.metadata?.original_duration_seconds || evidence?.metadata?.duration_seconds || 0).toFixed(2)} s</strong></div><div><span>Decoded input</span><strong>${escapeHtml(evidence?.metadata?.channels || "—")} ch · ${escapeHtml(evidence?.metadata?.original_sample_rate || "—")} Hz</strong></div>`
        : `<div><span>Indicators</span><strong>18 structured inputs</strong></div><div><span>D/A/S estimates</span><strong>${evidence?.scores ? "Available" : "Unavailable"}</strong></div>`;
    return `<article class="report-card">
      <div class="report-card-head"><h3>${escapeHtml(modalityName)}</h3><strong>${percent(contribution?.weight)} fused weight</strong></div>
      <p>${detail}</p>
      <div class="report-meta">
        <div><span>Base reliability</span><strong>${percent(contribution?.base_reliability)}</strong></div>
        <div><span>Sample confidence</span><strong>${percent(contribution?.sample_confidence)}</strong></div>
        ${technical}
      </div>
      <h3>Stress evidence distribution</h3>
      ${reportProbabilities(evidence?.stress_probabilities)}
    </article>`;
  }

  async function buildClinicalReport() {
    if (!state.final) throw new Error("Run a multimodal assessment before generating the report.");
    const [config, health] = await Promise.all([
      configPromise.catch(() => ({ features: {} })),
      healthPromise.catch(() => ({ version: "Unavailable" })),
    ]);
    const result = state.final;
    const generated = new Date();
    const ranked = Object.entries(result.probabilities || {}).sort((first, second) => second[1] - first[1]);
    const winner = ranked[0] || [result.final_class, result.confidence];
    const runnerUp = ranked[1] || ["Unavailable", 0];
    const margin = Math.max(0, Number(winner[1]) - Number(runnerUp[1]));
    const contributions = Object.entries(result.contributions || {}).sort((first, second) => second[1].weight - first[1].weight);
    const strongest = contributions[0];
    const modalityCards = contributions.map(([name, contribution]) =>
      reportModalityCard(name, state[name], contribution)
    ).join("");
    const scoreCards = state.numerical?.scores
      ? Object.entries(state.numerical.scores).map(([name, item]) => `<div class="report-score">
          <span>Model-estimated ${escapeHtml(label(name))}</span>
          <strong>${Number(item.value).toFixed(1)} <small>/ ${Number(item.range[1]).toFixed(0)}</small></strong>
          <small>Raw model output: ${Number(item.raw_value).toFixed(2)} · presentation range ${item.range[0]}–${item.range[1]}</small>
        </div>`).join("")
      : `<div class="report-callout">D/A/S estimates were not produced because the numerical modality was not included.</div>`;
    const numericalValues = state.numerical
      ? numericalPayload(false)
      : {};
    const inputRows = Object.entries(numericalValues).map(([name, value]) => {
      const metadata = config.features?.[name] || {};
      return `<tr><td>${escapeHtml(label(name))}</td><td class="numeric">${escapeHtml(value)}</td><td>${escapeHtml(metadata.unit || "—")}</td><td class="numeric">${escapeHtml(metadata.min ?? "—")}–${escapeHtml(metadata.max ?? "—")}</td></tr>`;
    }).join("");
    const influenceRows = (state.numerical?.local_explanation || []).slice(0, 10).map(item => `<tr>
      <td>${escapeHtml(label(item.feature))}</td>
      <td class="numeric">${Number(item.value).toFixed(2)}</td>
      <td class="numeric">${Number(item.reference).toFixed(2)}</td>
      <td class="numeric">${Number(item.influence) >= 0 ? "+" : ""}${Number(item.influence).toFixed(3)}</td>
      <td>${escapeHtml(item.direction)}</td>
    </tr>`).join("");
    const modalityRows = contributions.map(([name, contribution]) => `<tr>
      <td>${escapeHtml(label(name))}</td>
      <td>${escapeHtml(label(contribution.predicted_class))}</td>
      <td class="numeric">${percent(contribution.base_reliability)}</td>
      <td class="numeric">${percent(contribution.sample_confidence)}</td>
      <td class="numeric"><strong>${percent(contribution.weight)}</strong></td>
    </tr>`).join("");
    const explanationFigures = [
      state.face?.face_explanation || state.face?.gradcam
        ? `<figure class="report-figure"><img src="${state.face.face_explanation || state.face.gradcam}" alt="Facial model explanation"><figcaption><strong>Facial explanation.</strong> ${escapeHtml(state.face.explanation_method || "Predicted-class sensitivity")} highlights regions that most affected the facial output. It is not an anatomical or causal map.</figcaption></figure>`
        : "",
      state.audio?.audio_explanation
        ? `<figure class="report-figure"><img src="${state.audio.audio_explanation}" alt="Audio waveform, log-Mel spectrogram and saliency"><figcaption><strong>Speech explanation.</strong> Waveform, standardized log-Mel representation, and predicted-class input-gradient saliency. Saliency indicates model sensitivity, not clinical causation.</figcaption></figure>`
        : "",
    ].filter(Boolean).join("");
    const rationale = `The engine selected <strong>${escapeHtml(label(winner[0]))}</strong> because its fused probability was the largest at <strong>${percent(winner[1])}</strong>. The next-highest category was ${escapeHtml(label(runnerUp[0]))} at ${percent(runnerUp[1])}, a margin of ${percent(margin)}. ${strongest ? `The largest normalized modality weight was assigned to ${escapeHtml(label(strongest[0]))} (${percent(strongest[1].weight)}), based on its held-out reliability and confidence of this sample.` : ""} These weights describe the fusion calculation; they are not causal or clinical-importance estimates.`;
    const qualityText = result.conflict
      ? "The conflict flag is active because the modality distributions show material disagreement. The combined result should be interpreted cautiously and each modality reviewed independently."
      : "No material cross-modality conflict was detected under the configured agreement rule. This does not establish clinical concordance or diagnostic certainty.";
    const report = document.getElementById("clinical-report");
    report.innerHTML = `
      <header class="report-header">
        <div class="report-brand"><div class="report-mark">M</div><div><h1>MindFuse XAI</h1><p>Multimodal Mental-Health Decision-Support Assessment</p></div></div>
        <div class="report-identifiers">
          <div><span>Report ID</span><strong>${escapeHtml(result.session_id || "Local session")}</strong></div>
          <div><span>Generated</span><strong>${escapeHtml(generated.toLocaleString())}</strong></div>
          <div><span>System version</span><strong>${escapeHtml(health.version || "Unavailable")}</strong></div>
          <div><span>Subject identifier</span><strong>Not collected</strong></div>
        </div>
      </header>
      <div class="report-banner"><strong>Experimental decision support — not a diagnosis or medical record.</strong> This report summarizes model output for professional review and must not replace a clinical interview, validated screening instrument, risk assessment, or qualified care.</div>
      <section class="report-result">
        <div class="primary"><span>Model-estimated stress category</span><strong>${escapeHtml(label(result.final_class))}</strong></div>
        <div><span>Top probability</span><strong>${percent(result.confidence)}</strong></div>
        <div><span>Agreement</span><strong>${percent(result.agreement_score)}</strong></div>
        <div><span>Uncertainty</span><strong>${percent(result.uncertainty_score)}</strong></div>
      </section>
      <section class="report-section">
        <h2>1. Executive interpretation and decision rationale</h2>
        <div class="report-rationale"><p>${rationale}</p><p><strong>Decision-quality context:</strong> ${qualityText}</p></div>
      </section>
      <section class="report-section">
        <h2>2. Fused probability distribution</h2>
        ${reportProbabilities(result.probabilities)}
        <p>The displayed category is the maximum of this normalized four-class distribution. All probabilities sum to 100%; none is a clinical prevalence or individual disease risk.</p>
      </section>
      <section class="report-section allow-break">
        <h2>3. Evidence by modality</h2>
        <div class="report-grid-2">${modalityCards}</div>
      </section>
      <section class="report-section">
        <h2>4. How the modalities influenced the result</h2>
        <table class="report-table"><thead><tr><th>Modality</th><th>Modality conclusion</th><th class="numeric">Held-out reliability</th><th class="numeric">Sample confidence</th><th class="numeric">Normalized weight</th></tr></thead><tbody>${modalityRows}</tbody></table>
        <p>The system multiplies held-out validation reliability by a bounded entropy-confidence factor, then normalizes across available modalities. The final distribution is the weighted average of their complete stress distributions—not a vote based only on top labels.</p>
      </section>
      <section class="report-section">
        <h2>5. Depression, Anxiety and Stress score estimates</h2>
        <div class="report-score-grid">${scoreCards}</div>
        <p><strong>Interpretation boundary:</strong> these are machine-learning estimates of organizer-dataset targets. They are not administered PHQ-9, GAD-7, DASS-21, or other validated questionnaire results, and no diagnostic thresholds are applied in this report.</p>
      </section>
      ${influenceRows ? `<section class="report-section allow-break"><h2>6. Numerical evidence rationale</h2>
        <table class="report-table"><thead><tr><th>Indicator</th><th class="numeric">Observed</th><th class="numeric">Training reference</th><th class="numeric">Severity shift</th><th>Model direction</th></tr></thead><tbody>${influenceRows}</tbody></table>
        <p>“Severity shift” is the change in expected ordinal model severity when this feature is replaced by its training reference while all other values remain fixed. It is local sensitivity, not causation, biological mechanism, or clinical effect size.</p></section>` : ""}
      ${inputRows ? `<section class="report-section allow-break"><h2>7. Structured behavioral and physiological inputs</h2>
        <table class="report-table"><thead><tr><th>Indicator</th><th class="numeric">Recorded value</th><th>Unit</th><th class="numeric">Accepted UI range</th></tr></thead><tbody>${inputRows}</tbody></table></section>` : ""}
      ${explanationFigures ? `<section class="report-section allow-break"><h2>8. Model explanation visuals</h2><div class="report-explanations">${explanationFigures}</div></section>` : ""}
      <section class="report-section">
        <h2>9. Professional review checklist</h2>
        <div class="report-grid-2">
          <div class="report-callout"><h3>Before using this output</h3><ul class="report-list"><li>Verify input identity, timing, recording quality, and contextual factors.</li><li>Review every modality distribution, not only the fused category.</li><li>Corroborate with clinical interview and appropriately validated instruments.</li><li>Document medication, sleep, acute illness, environment, and accessibility factors that may affect signals.</li></ul></div>
          <div class="report-callout warning"><h3>Safety and escalation</h3><ul class="report-list"><li>This system does not assess suicidality, self-harm, violence, psychosis, abuse, or medical emergency.</li><li>Conduct an independent urgent-risk assessment whenever clinically indicated.</li><li>Do not delay emergency services or qualified care because of a reassuring model result.</li><li>Do not use this output for autonomous diagnosis, triage, treatment, employment, insurance, or access decisions.</li></ul></div>
        </div>
      </section>
      <section class="report-section">
        <h2>10. Technical provenance and limitations</h2>
        <ul class="report-list"><li>Fusion method: ${escapeHtml(result.method || "confidence-aware weighted linear opinion pool")}.</li><li>Modalities included: ${escapeHtml(contributions.map(([name]) => label(name)).join(", "))}.</li><li>Training datasets were modality-specific and unpaired; models were trained independently and combined only for this live session.</li><li>Organizer data are limited: small face sample, missing documented speech class, and weak numerical target relationships. Performance may not generalize across populations, devices, languages, health conditions, or recording environments.</li><li>Uploaded media are processed under opaque temporary filenames and removed after request completion; this PDF may still contain sensitive derived information and should be handled accordingly.</li></ul>
      </section>
      <section class="report-section"><h2>11. Reviewer documentation</h2><p>Clinical/contextual observations:</p><div class="report-signoff"><div>Reviewer name and role</div><div>Review date and signature</div></div></section>
      <div class="report-footer">MindFuse XAI · Research prototype · ${escapeHtml(result.session_id || "Local session")} · Generated ${escapeHtml(generated.toISOString())}</div>`;
    report.setAttribute("aria-hidden", "false");
    return report;
  }

  function renderFinal(result) {
    const container = document.getElementById("fusion-result");
    document.getElementById("final-class").textContent = label(result.final_class);
    document.getElementById("final-confidence").textContent = percent(result.confidence);
    document.getElementById("confidence-ring").style.setProperty("--confidence", `${result.confidence * 360}deg`);
    const summary = result.conflict
      ? "Modalities show material disagreement; review each contribution."
      : "Available modalities were combined with calibrated, entropy-adjusted weights.";
    document.getElementById("final-summary").textContent = `${summary} · Session ${result.session_id || "local"}`;
    document.getElementById("fusion-probabilities").innerHTML = probabilityBars(result.probabilities);
    document.getElementById("fusion-contributions").innerHTML = Object.entries(result.contributions).map(([name, item]) =>
      `<div class="contribution-item"><span>${escapeHtml(label(name))} · ${escapeHtml(label(item.predicted_class))}</span><strong>${percent(item.weight)}</strong></div>`
    ).join("");
    document.getElementById("quality-metrics").innerHTML = [
      ["Agreement", percent(result.agreement_score)],
      ["Uncertainty", percent(result.uncertainty_score)],
      ["Conflict flag", result.conflict ? "Review" : "Clear"],
    ].map(([name, value]) => `<div class="quality-item"><span>${name}</span><strong>${value}</strong></div>`).join("");
    const scores = state.numerical?.scores;
    document.getElementById("score-results").innerHTML = scores
      ? Object.entries(scores).map(([name, item]) => `<article class="score-card"><span>Estimated ${escapeHtml(label(name))}</span><strong>${Number(item.value).toFixed(1)} <small>/ ${item.range[1]}</small></strong><i><b style="width:${item.value / item.range[1] * 100}%"></b></i></article>`).join("")
      : `<div class="notice">D/A/S scores unavailable: numerical indicators were not included.</div>`;
    const evidence = [];
    if (state.face?.face_explanation || state.face?.gradcam) {
      const image = state.face.face_explanation || state.face.gradcam;
      evidence.push(`<article class="evidence-card"><h3>Facial ${escapeHtml(state.face.explanation_method || "model explanation")} · ${escapeHtml(state.face.emotion)}</h3><img src="${image}" alt="Face model explanation"></article>`);
    }
    if (state.audio?.audio_explanation) {
      evidence.push(`<article class="evidence-card"><h3>Audio saliency · ${escapeHtml(state.audio.emotion)}</h3><img src="${state.audio.audio_explanation}" alt="Audio waveform, spectrogram, and saliency"></article>`);
    }
    if (state.numerical?.local_explanation) {
      evidence.push(`<article class="evidence-card"><h3>Strongest numerical indicators</h3>${state.numerical.local_explanation.slice(0, 8).map(item => `<div class="influence-row ${item.influence > 0 ? "risk" : "protective"}"><span>${escapeHtml(label(item.feature))}</span><strong>${item.influence > 0 ? "+" : ""}${Number(item.influence).toFixed(3)}</strong></div>`).join("")}</article>`);
    }
    document.getElementById("evidence-results").innerHTML = evidence.join("");
    container.hidden = false;
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("run-assessment").addEventListener("click", async event => {
    const button = event.currentTarget;
    setButtonLoading(button, true);
    try {
      if (faceInput.files[0] && !state.face) await analyzeFace();
      if (audioInput.files[0] && !state.audio) await analyzeAudio();
      const entered = [...form.elements].filter(element => element.name).some(element => element.value !== "");
      if (entered && !state.numerical) await analyzeNumerical();
      const modalities = {};
      if (state.face) modalities.face = state.face.stress_probabilities;
      if (state.audio) modalities.audio = state.audio.stress_probabilities;
      if (state.numerical) modalities.numerical = state.numerical.stress_probabilities;
      if (!Object.keys(modalities).length) throw new Error("Add at least one modality before running the assessment.");
      const result = await api("/api/predict/multimodal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modalities }),
      });
      state.final = {
        ...result,
        scores: state.numerical?.scores || null,
        evidence: { face: state.face, audio: state.audio, numerical: state.numerical },
      };
      renderFinal(state.final);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setButtonLoading(button, false);
    }
  });

  document.getElementById("reset-assessment").addEventListener("click", () => {
    faceInput.value = "";
    audioInput.value = "";
    form.reset();
    Object.keys(state).forEach(key => { state[key] = null; });
    Object.keys(revisions).forEach(key => { revisions[key] += 1; });
    document.querySelectorAll(".inline-result, #fusion-result").forEach(node => { node.hidden = true; });
    document.getElementById("face-preview").hidden = true;
    document.getElementById("audio-error").hidden = true;
    document.getElementById("audio-file-meta").textContent = "Mono/stereo WAV · speech content";
    document.getElementById("profile-loaded").textContent = "";
    document.querySelectorAll("[data-profile]").forEach(item => item.classList.remove("active"));
    document.getElementById("analyze-face").disabled = true;
    document.getElementById("analyze-audio").disabled = true;
    ["face", "audio", "numerical"].forEach(name => setState(name, "", "Optional"));
    toast("Assessment cleared.");
  });

  document.getElementById("download-json").addEventListener("click", () => {
    if (!state.final) return;
    const blob = new Blob([JSON.stringify({ generated_at: new Date().toISOString(), ...state.final }, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `mindfuse-assessment-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  });
  document.getElementById("print-report").addEventListener("click", async event => {
    const button = event.currentTarget;
    const originalTitle = document.title;
    setButtonLoading(button, true);
    try {
      await buildClinicalReport();
      document.title = `MindFuse-${state.final.session_id || "assessment-report"}`;
      window.addEventListener("afterprint", () => { document.title = originalTitle; }, { once: true });
      window.print();
    } catch (error) {
      document.title = originalTitle;
      toast(error.message, "error");
    } finally {
      setButtonLoading(button, false);
    }
  });
  window.MindFuse.buildClinicalReport = buildClinicalReport;
});

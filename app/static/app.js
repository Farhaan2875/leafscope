"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";

const S = { status: null, result: null, layer: null, truth: null, busy: false };
const KIND = { fungal: "Fungal", oomycete: "Oomycete", bacterial: "Bacterial", viral: "Viral", mite: "Mite", healthy: "Healthy", unknown: "Unknown" };

/* ------------------------------ header lamps ------------------------------ */
async function loadStatus() {
  const r = await fetch("/api/status");
  S.status = await r.json();
  const lamps = $("#lamps");
  if (!S.status.examiners.length) {
    lamps.innerHTML = `<span class="lamp off"><i></i>No examiners trained yet</span>`;
  } else {
    lamps.innerHTML = S.status.examiners.map((e) =>
      `<span class="lamp"><i></i>${esc(e.label)} · ${esc(e.device)}</span>`).join("");
  }
}

/* ----------------------------- specimen drawer ---------------------------- */
async function loadDrawer() {
  const box = $("#specimens");
  const r = await fetch("/api/samples?n=24");
  const list = await r.json();
  if (!list.length) {
    box.innerHTML = `<p class="fine">Run <code>prepare_data.py</code> to fill the drawer with real test leaves.</p>`;
    return;
  }
  box.innerHTML = list.map((s) =>
    `<button class="spec" data-id="${s.id}" aria-label="Open specimen ${esc(s.accession)}">
       <img loading="lazy" src="/api/sample/${s.id}" alt=""><span>${esc(s.accession)}</span></button>`).join("");
  box.addEventListener("click", async (e) => {
    const b = e.target.closest(".spec");
    if (!b || S.busy) return;
    box.querySelectorAll(".spec").forEach((x) => x.removeAttribute("aria-current"));
    b.setAttribute("aria-current", "true");
    const resp = await fetch(`/api/sample/${b.dataset.id}`);
    const truth = JSON.parse(atob(resp.headers.get("X-Truth")));
    examine(await resp.blob(), { name: `Specimen PV-${String(+b.dataset.id + 1).padStart(3, "0")}`, truth });
  });
}

/* ------------------------------- drop / paste ------------------------------ */
const drop = $("#drop"), file = $("#file");
drop.addEventListener("click", (e) => { e.preventDefault(); file.click(); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) fromFile(f); });
file.addEventListener("change", () => { if (file.files[0]) fromFile(file.files[0]); file.value = ""; });
window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
  if (f) fromFile(f);
});
function fromFile(f) {
  document.querySelectorAll(".spec").forEach((x) => x.removeAttribute("aria-current"));
  examine(f, { name: f.name || "Pasted image", truth: null });
}

/* ------------------------------- examination ------------------------------- */
async function examine(blob, meta) {
  if (S.busy) return;
  S.busy = true;
  S.truth = meta.truth;
  const base = $("#img-base"), over = $("#img-over");
  $("#stage").classList.remove("empty");
  base.hidden = false;
  base.src = URL.createObjectURL(blob);
  over.hidden = true;
  $("#curtain").hidden = true; $("#over-tag").hidden = true; $("#rulebar").hidden = true;
  $("#layers").innerHTML = "";
  $("#scanline").hidden = false;
  $("#caption").textContent = `Fig. — ${meta.name} · examining…`;
  $("#ledger").innerHTML = `<div class="ledger-idle"><p class="serif-lg">Examining…</p><p>Segmenting the leaf, measuring lesions, and asking both examiners.</p></div>`;
  try {
    const resp = await fetch("/api/diagnose?model=both", { method: "POST", headers: { "Content-Type": blob.type || "application/octet-stream" }, body: blob });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Request failed");
    S.result = data;
    renderPlate(data, meta);
    renderLedger(data, meta);
  } catch (err) {
    $("#ledger").innerHTML = `<div class="ledger-idle"><p class="serif-lg error">Could not examine that file.</p><p>${esc(err.message)}</p></div>`;
    $("#caption").textContent = "Fig. — examination failed";
  } finally {
    $("#scanline").hidden = true;
    S.busy = false;
  }
}

/* ---------------------------------- plate --------------------------------- */
function renderPlate(r, meta) {
  const stage = $("#stage");
  stage.style.aspectRatio = `${r.size[0]} / ${r.size[1]}`;
  $("#img-base").src = r.image;
  const layers = [];
  Object.entries(r.examiners).forEach(([k, e]) => layers.push({ id: k, label: `Gaze · ${e.label}`, tag: `${e.label} attention`, src: e.cam }));
  layers.push({ id: "lesions", label: "Lesion map", tag: "Lesions (OpenCV)", src: r.severity.overlay });
  S.layers = layers;
  $("#layers").innerHTML = layers.map((l) => `<button class="layer" data-id="${esc(l.id)}" aria-pressed="false">${esc(l.label)}</button>`).join("");
  $("#layers").onclick = (e) => { const b = e.target.closest(".layer"); if (b) setLayer(b.dataset.id); };
  setLayer(layers[0].id);
  $("#rulebar").hidden = false;
  $("#caption").textContent = `Fig. — ${meta.name} · ${r.size[0]}×${r.size[1]} px`;
}
function setLayer(id) {
  const l = S.layers.find((x) => x.id === id);
  S.layer = id;
  const over = $("#img-over");
  over.src = l.src; over.hidden = false;
  $("#curtain").hidden = false;
  $("#over-tag").hidden = false; $("#over-tag").textContent = l.tag;
  $("#rule-layer").textContent = l.label;
  document.querySelectorAll(".layer").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  setReveal(+$("#reveal").value);
}
function setReveal(v) {
  v = Math.max(0, Math.min(100, v));
  $("#reveal").value = v;
  $("#img-over").style.clipPath = `inset(0 0 0 ${v}%)`;
  $("#curtain").style.left = v + "%";
}
$("#reveal").addEventListener("input", (e) => setReveal(+e.target.value));
(() => {
  const stage = $("#stage"); let drag = false;
  const move = (e) => { const b = stage.getBoundingClientRect(); setReveal(((e.clientX - b.left) / b.width) * 100); };
  stage.addEventListener("pointerdown", (e) => { if (stage.classList.contains("empty") || $("#curtain").hidden) return; drag = true; stage.setPointerCapture(e.pointerId); move(e); });
  stage.addEventListener("pointermove", (e) => { if (drag) move(e); });
  stage.addEventListener("pointerup", () => (drag = false));
  stage.addEventListener("pointercancel", () => (drag = false));
})();

/* --------------------------------- ledger --------------------------------- */
function ticks(p) {
  const on = Math.round(p * 25);
  return `<div class="ticks ${p < 0.6 ? "low" : ""}" role="img" aria-label="confidence ${pct(p, 0)}">` +
    Array.from({ length: 25 }, (_, i) => `<i class="${i < on ? "on" : ""}"></i>`).join("") + `</div>`;
}
function markerLeft(percent) {
  // five equal-width grade bands; position is linear inside each band
  const edges = [0, 1, 10, 25, 50, 100];
  for (let i = 0; i < 5; i++) if (percent <= edges[i + 1]) return i * 20 + ((percent - edges[i]) / (edges[i + 1] - edges[i])) * 20;
  return 100;
}
function gazeBar(a) {
  return `<div class="gaze"><span class="k">Where it looked</span>
    <div class="gaze-bar" role="img" aria-label="attention: ${a.lesion}% lesion, ${a.leaf}% healthy leaf, ${a.background}% background">
      <i style="width:${a.lesion}%"></i><i style="width:${a.leaf}%"></i><i style="width:${a.background}%"></i></div>
    <div class="gaze-key"><span><b style="background:var(--rust)"></b>lesion ${a.lesion}%</span><span><b style="background:var(--moss)"></b>leaf ${a.leaf}%</span><span><b style="background:var(--rule)"></b>background ${a.background}%</span></div></div>`;
}

function renderLedger(r, meta) {
  const ex = Object.entries(r.examiners);
  const sev = r.severity;
  let html = "";
  (r.notes || []).forEach((n) => (html += `<p class="notice">${esc(n)}</p>`));

  // ---- Finding
  let lead = null;
  if (ex.length) lead = ex.map(([, e]) => e.top[0]).sort((a, b) => b.prob - a.prob)[0];
  if (lead) {
    html += `<div class="row"><span class="k">Finding</span>
      <div class="finding-crop">${esc(lead.crop)}</div>
      <div class="finding-name ${lead.healthy ? "healthy" : ""}">${esc(lead.disease)}</div>
      ${lead.pathogen ? `<div class="pathogen">${esc(lead.pathogen)}<span class="chip">${esc(KIND[lead.kind] || lead.kind)}</span></div>` : ""}
      ${r.agree === false ? `<p class="disagree-note">The examiners disagree. This is the more confident opinion (${pct(lead.prob, 0)}).</p>` : ""}
      </div>`;
  } else {
    html += `<div class="row"><span class="k">Finding</span><div class="finding-name">Examiners offline</div>
      <p class="method">Train the models first (<code>train.py</code>). The lesion measurement below works without them.</p></div>`;
  }

  // ---- Severity
  const healthy = lead && lead.healthy;
  html += `<div class="row"><span class="k">Severity · lesion area ÷ leaf area</span>
    <div class="sev-top"><div class="sev-pct" id="sev-num">0<small>%</small></div>
    <div class="sev-grade">${healthy ? "<b>Not applicable</b><br>leaf read as healthy" : `Grade <b>${sev.grade}</b> of 4<br>${esc(sev.label)}`}</div></div>
    <div class="ruler ${healthy ? "dim" : ""}"><span></span><span></span><span></span><span></span><span></span><i class="marker" id="marker" style="left:0%"></i></div>
    <div class="ruler-labels"><span>0</span><span>1</span><span>10</span><span>25</span><span>50</span><span>100 %</span></div>
    <p class="method">${healthy
      ? "Colour changes flagged on a healthy leaf are usually veins, glare or texture, not disease."
      : "Estimated by OpenCV: the leaf is segmented, then tissue that departs from its healthy-green colour is counted as lesion. Large grey dead patches can be under-counted."}</p></div>`;

  // ---- Examiners
  if (ex.length) {
    html += `<div class="row"><span class="k">Two examiners</span>`;
    ex.forEach(([k, e]) => {
      const t = e.top[0];
      html += `<div class="examiner"><div class="ex-head"><span class="ex-name">${esc(e.label)}</span><span class="ex-ms">${e.ms} ms</span></div>
        <div class="ex-verdict">${esc(t.crop)} — ${esc(t.disease)}</div>${ticks(t.prob)}
        <div class="conf"><span>confidence</span><span>${pct(t.prob)}</span></div>
        <ul class="alts">${e.top.slice(1).map((a) => `<li><span>${esc(a.crop)} — ${esc(a.disease)}</span><span>${pct(a.prob)}</span></li>`).join("")}</ul>
        ${gazeBar(e.attention)}</div>`;
    });
    if (ex.length > 1) html += `<span class="stamp ${r.agree ? "ok" : "split"}">${r.agree ? "Concur" : "Split opinion"}</span>`;
    html += `</div>`;
  }

  // ---- Field note
  if (lead) html += `<div class="row"><span class="k">Field note</span><p class="advice">${esc(lead.advice)}</p></div>`;

  // ---- Sealed record (specimens only)
  if (meta.truth) {
    html += `<div class="row" id="truth-row"><span class="k">Dataset record</span>
      <button class="seal" id="seal">Sealed. Break the seal to compare with the label on file.</button></div>`;
  }
  $("#ledger").innerHTML = html;

  // count-up + marker
  const target = sev.percent, t0 = performance.now();
  const num = $("#sev-num");
  requestAnimationFrame(() => ($("#marker").style.left = markerLeft(target) + "%"));
  (function tick(now) {
    const p = Math.min(1, (now - t0) / 650);
    num.innerHTML = `${(target * (1 - Math.pow(1 - p, 3))).toFixed(1)}<small>%</small>`;
    if (p < 1) requestAnimationFrame(tick);
  })(t0);

  const seal = $("#seal");
  if (seal) seal.addEventListener("click", () => {
    const tr = meta.truth;
    const ok = lead && tr.class_name === lead.class_name;
    $("#truth-row").innerHTML = `<span class="k">Dataset record</span><div class="truth">
      <div><div class="finding-crop">${esc(tr.crop)}</div><div class="t-name">${esc(tr.disease)}</div></div>
      <span class="stamp ${ok ? "ok" : "split"}">${ok ? "Matches" : "Differs"}</span></div>`;
  });
}

/* --------------------------------- trials --------------------------------- */
const fmt = {
  pct: (v) => (v * 100).toFixed(2) + "%", f: (v, d = 1) => Number(v).toFixed(d),
};
async function renderTrials() {
  const body = $("#trials-body");
  const r = await fetch("/api/trials");
  const d = await r.json();
  const keys = Object.keys(d.models || {});
  if (!keys.length) {
    body.innerHTML = `<div class="empty-trials"><p class="serif-lg">No trial results yet.</p>
      <p>Train both models, then run <code>python evaluate.py</code>. This page fills in automatically.</p></div>`;
    return;
  }
  const M = d.models, [a, b] = keys;
  const rows = [
    ["Test accuracy", (m) => m.test.accuracy, fmt.pct, "max"],
    ["Macro-F1", (m) => m.test.macro_f1, (v) => v.toFixed(4), "max"],
    ["Weighted F1", (m) => m.test.weighted_f1, (v) => v.toFixed(4), "max"],
    ["Parameters (M)", (m) => m.params_million, (v) => fmt.f(v, 2), "min"],
    ["Model size (MB)", (m) => m.size_mb, (v) => fmt.f(v, 1), "min"],
    ["Latency, batch 1 (ms)", (m) => m.latency_ms_batch1, (v) => fmt.f(v, 1), "min"],
    ["Training time (min)", (m) => m.train_minutes, (v) => fmt.f(v, 1), "min"],
    ["Attention on lesions", (m) => m.gradcam.attention_on_lesion_pct, (v) => fmt.f(v, 1) + "%", "max"],
    ["Attention on background", (m) => m.gradcam.attention_on_background_pct, (v) => fmt.f(v, 1) + "%", "min"],
  ];
  let html = `<table class="ledger-table"><thead><tr><th>Measure</th>${keys.map((k) => `<th>${esc(M[k].label)}</th>`).join("")}</tr></thead><tbody>`;
  rows.forEach(([name, get, f, best]) => {
    const vals = keys.map((k) => get(M[k]));
    const win = keys.length > 1 ? (best === "max" ? Math.max(...vals) : Math.min(...vals)) : null;
    html += `<tr><td class="metric">${esc(name)}</td>${vals.map((v) => `<td class="num ${win !== null && v === win && vals[0] !== vals[1] ? "better" : ""}">${f(v)}</td>`).join("")}</tr>`;
  });
  html += `</tbody></table>`;

  const variants = Object.keys(M[a].lighting);
  html += `<h3 class="section-title">Under changing light</h3>
    <p class="section-note">The same test photographs, darkened, over-exposed, half-shadowed or colour-cast with OpenCV before classification. Bars show macro-F1; a model that holds up here is less likely to be thrown by a bright field.</p>
    <div class="light-grid">`;
  variants.forEach((v) => {
    html += `<div class="name">${esc(v)}</div><div class="bars">` + keys.map((k, i) => {
      const x = M[k].lighting[v].macro_f1;
      return `<div class="bar-line"><div class="bar-track"><div class="bar-fill ${i ? "b" : "a"}" style="width:${(x * 100).toFixed(1)}%"></div></div><span>${(x * 100).toFixed(1)}</span></div>`;
    }).join("") + `</div>`;
  });
  html += `</div><div class="legend">${keys.map((k, i) => `<span><b style="background:var(${i ? "--ochre" : "--moss"})"></b>${esc(M[k].label)}</span>`).join("")}</div>`;

  html += `<h3 class="section-title">Figures</h3><div class="figs">`;
  const figs = [["comparison.png", "Headline comparison"]];
  keys.forEach((k) => { figs.push([`training_${k}.png`, `${M[k].label} · training curves`], [`confusion_${k}.png`, `${M[k].label} · confusion matrix`], [`gradcam_gallery_${k}.png`, `${M[k].label} · Grad-CAM, correct vs misclassified`]); });
  figs.forEach(([f, c]) => (html += `<figure class="fig"><a href="/figures/${f}" target="_blank" rel="noopener"><img loading="lazy" src="/figures/${f}" alt="${esc(c)}" onerror="this.closest('figure').remove()"></a><figcaption>${esc(c)}</figcaption></figure>`));
  html += `</div>`;
  body.innerHTML = html;
}

/* ------------------------------- navigation ------------------------------- */
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.setAttribute("aria-pressed", String(x === t)));
  const v = t.dataset.view;
  $("#view-bench").hidden = v !== "bench";
  $("#view-trials").hidden = v !== "trials";
  if (v === "trials") renderTrials();
}));

loadStatus();
loadDrawer();

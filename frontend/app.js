// SignSpeak web app: live camera + photo recognition, message composer and speech.
// Everything runs in the browser: MediaPipe finds the hand joints and classifier.js runs the
// trained network on them. The optional Python server adds the Azure voice and grammar correction.
import { SignModel } from "./classifier.js";

const MP_VERSION = "1.0.1";
const MP_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_URL = "models/hand_landmarker.task";
const SIGN_MODEL_URL = "models/sign_mlp.json";

const PREDICT_INTERVAL_MS = 90;   // classify at most ~11 times per second
const AUTO_SPACE_MS = 1400;       // hands lowered this long -> word break
const SMOOTHING = 0.45;           // weight of the newest prediction in the moving average

// MediaPipe hand skeleton (pairs of landmark indices), used to draw photo overlays.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const $ = (id) => document.getElementById(id);
const el = {
  status: $("serverStatus"), themeToggle: $("themeToggle"),
  tabCamera: $("tabCamera"), tabPhotos: $("tabPhotos"), panelCamera: $("panelCamera"), panelPhotos: $("panelPhotos"),
  video: $("video"), overlay: $("overlay"), stageEmpty: $("stageEmpty"), handsChip: $("handsChip"),
  btnCameraStart: $("btnCameraStart"), btnCameraStop: $("btnCameraStop"),
  dropzone: $("dropzone"), fileInput: $("fileInput"), photoStrip: $("photoStrip"), photoActions: $("photoActions"),
  photoText: $("photoText"), btnPhotosClear: $("btnPhotosClear"), btnPhotosRecognize: $("btnPhotosRecognize"), btnPhotosUse: $("btnPhotosUse"),
  holdRing: $("holdRing"), predLetter: $("predLetter"), predHint: $("predHint"), topList: $("topList"),
  message: $("message"), voiceLabel: $("voiceLabel"), audio: $("audio"),
  btnSpace: $("btnSpace"), btnBack: $("btnBack"), btnClear: $("btnClear"), btnCopy: $("btnCopy"), btnDownload: $("btnDownload"),
  btnGrammar: $("btnGrammar"), btnSpeak: $("btnSpeak"),
  holdTime: $("holdTime"), holdTimeOut: $("holdTimeOut"), minConf: $("minConf"), minConfOut: $("minConfOut"), autoSpace: $("autoSpace"),
  guideGrid: $("guideGrid"), modelInfo: $("modelInfo"), toasts: $("toasts"),
};

const state = {
  model: null,
  server: null,  // /api/health of the optional Python server
  settings: { holdMs: 1000, minConf: 0.7, autoSpace: true },
  cam: { running: false, stream: null, landmarker: null, mp: null, drawer: null, raf: 0, lastVideoTime: -1, lastSent: 0 },
  live: { ema: new Map(), candidate: null, since: 0, locked: null, lastHandAt: 0, typedSinceSpace: false, showing: false },
  photos: [],
  nextPhotoId: 1,
};

// ── Small helpers ───────────────────────────────────────────────────────────

const storage = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode etc. */ } },
};

async function api(path, options = {}) {
  const res = await fetch(path, options);  // relative paths, so the app also works under a sub-path
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const body = await res.json(); if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch {}
    throw new Error(detail);
  }
  return res;
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(text, { kind = "info", action = null, timeout = 3500 } = {}) {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  const span = document.createElement("span");
  span.textContent = text;
  node.append(span);
  if (action) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = action.label;
    b.onclick = () => { action.run(); node.remove(); };
    node.append(b);
  }
  el.toasts.append(node);
  setTimeout(() => node.remove(), timeout);
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
}

const pct = (p) => `${Math.round(p * 100)}%`;

// ── Server status ───────────────────────────────────────────────────────────

async function loadModel() {
  try {
    state.model = await SignModel.load(SIGN_MODEL_URL);
    const classes = state.model.classes;
    el.status.dataset.state = "ok";
    el.status.querySelector(".label").textContent = `Model ready · ${classes.length} signs`;
    const m = state.model.metrics;
    if (m.test_accuracy) el.modelInfo.textContent = `Hand-landmark model · ${classes.length} ISL signs · ${(m.test_accuracy * 100).toFixed(2)}% on held-out dataset frames`;
    renderGuide(classes);
  } catch (err) {
    console.error(err);
    el.status.dataset.state = "error";
    el.status.querySelector(".label").textContent = "Model failed to load";
    el.btnCameraStart.disabled = true;
    el.btnPhotosRecognize.disabled = true;
  }
}

// The Python server is optional: when it runs, it can offer the Azure voice and grammar correction.
async function detectServer() {
  try {
    const health = await (await api("api/health")).json();
    if (health.status !== "ok") return;
    state.server = health;
    el.btnGrammar.hidden = !health.features.grammar;
    if (health.features.azure_tts) el.voiceLabel.textContent = "Voice: Azure neural (en-IN)";
  } catch {
    // static hosting (e.g. GitHub Pages): browser voice, no grammar correction
  }
}

// ── Tabs & theme ────────────────────────────────────────────────────────────

function selectTab(which) {
  const camera = which === "camera";
  el.tabCamera.setAttribute("aria-selected", String(camera));
  el.tabPhotos.setAttribute("aria-selected", String(!camera));
  el.tabCamera.tabIndex = camera ? 0 : -1;
  el.tabPhotos.tabIndex = camera ? -1 : 0;
  el.panelCamera.hidden = !camera;
  el.panelPhotos.hidden = camera;
  if (!camera) stopCamera();
}
el.tabCamera.onclick = () => selectTab("camera");
el.tabPhotos.onclick = () => selectTab("photos");
for (const tab of [el.tabCamera, el.tabPhotos]) {
  tab.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      const next = tab === el.tabCamera ? el.tabPhotos : el.tabCamera;
      next.focus();
      next.click();
    }
  });
}

el.themeToggle.onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("signspeak.theme", root.dataset.theme); } catch {}
};

// ── Live camera ─────────────────────────────────────────────────────────────

// MediaPipe is loaded once and shared by the camera (VIDEO mode) and photo (IMAGE mode) detectors.
let visionPromise = null;
function loadVision() {
  visionPromise ??= (async () => {
    const mp = await import(MP_BUNDLE);
    const fileset = await mp.FilesetResolver.forVisionTasks(MP_WASM);
    return { mp, fileset };
  })().catch((err) => { visionPromise = null; throw err; });
  return visionPromise;
}

async function createLandmarker(runningMode) {
  const { mp, fileset } = await loadVision();
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode,
    numHands: 2,
    minHandDetectionConfidence: runningMode === "IMAGE" ? 0.3 : 0.5,
    minHandPresenceConfidence: runningMode === "IMAGE" ? 0.3 : 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await mp.HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await mp.HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

async function createVideoLandmarker() {
  const landmarker = await createLandmarker("VIDEO");
  // Run once on a blank frame so shader compilation happens now, not on the user's first sign.
  const warmup = document.createElement("canvas");
  warmup.width = warmup.height = 64;
  warmup.getContext("2d").fillRect(0, 0, 64, 64);
  landmarker.detectForVideo(warmup, performance.now());
  state.cam.mp = (await loadVision()).mp;
  return landmarker;
}

function cameraErrorMessage(err) {
  if (!window.isSecureContext) return "The camera needs https:// or http://localhost.";
  switch (err?.name) {
    case "NotAllowedError": return "Camera permission was denied. Allow it in the browser's address bar and try again.";
    case "NotFoundError": return "No camera was found on this device.";
    case "NotReadableError": return "The camera is being used by another app.";
    default: return `Could not start the hand tracker: ${err?.message || err}`;
  }
}

async function startCamera() {
  const cam = state.cam;
  setBusy(el.btnCameraStart, true);
  el.btnCameraStart.textContent = "Loading hand tracker…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error("Camera API not available"), { name: "NotSupported" });
    if (!cam.landmarker) cam.landmarker = await createVideoLandmarker();
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
    el.video.srcObject = cam.stream;
    await el.video.play();
    el.overlay.width = el.video.videoWidth;
    el.overlay.height = el.video.videoHeight;
    cam.drawer = new cam.mp.DrawingUtils(el.overlay.getContext("2d"));
    cam.running = true;
    cam.lastVideoTime = -1;
    el.stageEmpty.hidden = true;
    el.handsChip.hidden = false;
    el.btnCameraStop.hidden = false;
    cam.raf = requestAnimationFrame(frameLoop);
  } catch (err) {
    console.error(err);
    stopCamera();
    toast(cameraErrorMessage(err), { kind: "error", timeout: 6000 });
  } finally {
    setBusy(el.btnCameraStart, false);
    el.btnCameraStart.textContent = "Start camera";
  }
}

function stopCamera() {
  const cam = state.cam;
  cam.running = false;
  cancelAnimationFrame(cam.raf);
  cam.stream?.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  el.video.srcObject = null;
  el.overlay.getContext("2d").clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.stageEmpty.hidden = false;
  el.handsChip.hidden = true;
  el.btnCameraStop.hidden = true;
  resetLive();
}

function frameLoop() {
  const cam = state.cam;
  if (!cam.running) return;
  if (el.video.readyState >= 2 && el.video.currentTime !== cam.lastVideoTime) {
    cam.lastVideoTime = el.video.currentTime;
    const result = cam.landmarker.detectForVideo(el.video, performance.now());
    drawHands(result.landmarks);
    onFrame(result.landmarks);
  }
  cam.raf = requestAnimationFrame(frameLoop);
}

function drawHands(hands) {
  const { drawer, mp } = state.cam;
  const ctx = el.overlay.getContext("2d");
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#0f766e";
  for (const hand of hands) {
    drawer.drawConnectors(hand, mp.HandLandmarker.HAND_CONNECTIONS, { color: accent, lineWidth: 4 });
    drawer.drawLandmarks(hand, { color: "#ffffff", fillColor: accent, lineWidth: 1, radius: 4 });
  }
  el.handsChip.textContent = hands.length ? `${hands.length} hand${hands.length > 1 ? "s" : ""}` : "No hands";
}

function onFrame(hands) {
  const now = performance.now();
  const cam = state.cam;
  const live = state.live;
  if (!hands.length) {
    onNoHands(now);
    return;
  }
  live.lastHandAt = now;
  if (!state.model || now - cam.lastSent < PREDICT_INTERVAL_MS) return;
  cam.lastSent = now;
  const pred = state.model.predict(hands.map((h) => h.map((p) => [p.x, p.y, p.z])), el.video.videoWidth, el.video.videoHeight, 5);
  onPrediction(pred, now);
}

function onPrediction(pred, now) {
  const live = state.live;
  const { minConf, holdMs } = state.settings;

  // Exponential moving average over class probabilities smooths out single-frame flicker.
  for (const [k, v] of live.ema) live.ema.set(k, v * (1 - SMOOTHING));
  for (const t of pred.top) live.ema.set(t.label, (live.ema.get(t.label) || 0) + SMOOTHING * t.prob);
  const ranked = [...live.ema].sort((a, b) => b[1] - a[1]);
  for (const [k, v] of ranked.slice(5)) if (v < 0.01) live.ema.delete(k);
  const [label, conf] = ranked[0];

  renderPrediction(label, conf, ranked.slice(0, 3));

  if (conf < minConf) {
    live.candidate = null;
    setHold(0, false, `Not sure yet (${pct(conf)})`);
    return;
  }
  if (label !== live.candidate) {
    live.candidate = label;
    live.since = now;
  }
  if (live.locked === label) {
    setHold(1, true, "Typed. Lower your hands to repeat it");
    return;
  }
  const progress = Math.min((now - live.since) / holdMs, 1);
  setHold(progress, false, "Hold steady…");
  if (progress >= 1) {
    typeText(label.toLowerCase());
    live.locked = label;
    live.typedSinceSpace = true;
    setHold(1, true, "Typed");
    el.holdRing.classList.remove("pulse");
    void el.holdRing.offsetWidth;
    el.holdRing.classList.add("pulse");
    highlightGuide(label);
  }
}

function onNoHands(now) {
  const live = state.live;
  if (live.showing) resetLive();
  if (state.settings.autoSpace && live.typedSinceSpace && now - live.lastHandAt > AUTO_SPACE_MS) {
    addSpace();
    live.typedSinceSpace = false;
  }
}

function resetLive() {
  const live = state.live;
  live.ema.clear();
  live.candidate = null;
  live.locked = null;
  live.showing = false;
  renderPrediction(null);
}

// ── Prediction panel ────────────────────────────────────────────────────────

function renderPrediction(label, conf = 0, top = []) {
  state.live.showing = label != null;
  el.predLetter.textContent = label ?? "–";
  el.topList.replaceChildren(...top.map(([lbl, p]) => {
    const li = document.createElement("li");
    const a = document.createElement("span"); a.className = "lbl"; a.textContent = lbl;
    const bar = document.createElement("span"); bar.className = "bar";
    const fill = document.createElement("span"); fill.style.width = pct(Math.min(p, 1)); bar.append(fill);
    const b = document.createElement("span"); b.className = "pct"; b.textContent = pct(Math.min(p, 1));
    li.append(a, bar, b);
    return li;
  }));
  if (label == null) setHold(0, false, state.cam.running ? "Show a sign to the camera" : "Waiting for a hand");
}

function setHold(progress, committed, hint) {
  el.holdRing.style.setProperty("--progress", progress.toFixed(3));
  el.holdRing.classList.toggle("committed", committed);
  if (hint) el.predHint.textContent = hint;
}

// ── Message ─────────────────────────────────────────────────────────────────

function typeText(text) {
  el.message.value += text;
  el.message.scrollTop = el.message.scrollHeight;
}
function addSpace() {
  if (el.message.value && !el.message.value.endsWith(" ")) typeText(" ");
}
function backspace() {
  el.message.value = el.message.value.slice(0, -1);
}

let undoText = null;
el.btnSpace.onclick = addSpace;
el.btnBack.onclick = backspace;
el.btnClear.onclick = () => {
  if (!el.message.value) return;
  undoText = el.message.value;
  el.message.value = "";
  toast("Message cleared", { action: { label: "Undo", run: () => { el.message.value = undoText; } } });
};
el.btnCopy.onclick = async () => {
  try { await navigator.clipboard.writeText(el.message.value); toast("Copied to clipboard"); }
  catch { toast("Copy failed. Select the text and copy it manually.", { kind: "error" }); }
};
el.btnDownload.onclick = () => {
  const text = el.message.value.trim();
  if (!text) return toast("Nothing to save yet");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text + "\n"], { type: "text/plain" }));
  a.download = `signspeak-${stamp}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

el.btnGrammar.onclick = async () => {
  const text = el.message.value.trim();
  if (!text) return toast("Nothing to correct yet");
  setBusy(el.btnGrammar, true);
  try {
    const out = await (await postJSON("api/grammar", { text })).json();
    if (out.text && out.text !== text) {
      undoText = el.message.value;
      el.message.value = out.text;
      toast("Grammar corrected", { action: { label: "Undo", run: () => { el.message.value = undoText; } } });
    } else {
      toast("Looks good already");
    }
  } catch (err) {
    toast(err.message, { kind: "error" });
  } finally {
    setBusy(el.btnGrammar, false);
  }
};

el.btnSpeak.onclick = speak;
async function speak() {
  const text = el.message.value.trim();
  if (!text) return toast("Sign or type something to speak");
  if (state.server?.features.azure_tts && !state.azureFailed) {
    setBusy(el.btnSpeak, true);
    try {
      const blob = await (await postJSON("api/speak", { text })).blob();
      if (el.audio.src) URL.revokeObjectURL(el.audio.src);
      el.audio.src = URL.createObjectURL(blob);
      await el.audio.play();
      return;
    } catch (err) {
      console.warn("Azure TTS failed, falling back to browser voice:", err);
      state.azureFailed = true;  // don't retry on every click
      el.voiceLabel.textContent = "Voice: browser (cloud voice unavailable)";
      toast("Cloud voice unavailable, using the browser voice");
    } finally {
      setBusy(el.btnSpeak, false);
    }
  }
  browserSpeak(text);
}

function browserSpeak(text) {
  if (!("speechSynthesis" in window)) return toast("This browser cannot speak text", { kind: "error" });
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voices = speechSynthesis.getVoices();
  utterance.voice = voices.find((v) => v.lang === "en-IN") || voices.find((v) => v.lang?.startsWith("en")) || null;
  utterance.lang = utterance.voice?.lang || "en-IN";
  speechSynthesis.speak(utterance);
}

document.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === " " && tag !== "BUTTON") { e.preventDefault(); addSpace(); }
  else if (e.key === "Backspace") { e.preventDefault(); backspace(); }
  else if (e.key === "Enter" && tag !== "BUTTON") { e.preventDefault(); speak(); }
});

// ── Photos ──────────────────────────────────────────────────────────────────

function addPhotos(fileList) {
  const files = [...fileList]
    .filter((f) => f.type.startsWith("image/"))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (!files.length) return toast("Those files are not images", { kind: "error" });
  for (const file of files) {
    state.photos.push({ id: state.nextPhotoId++, file, url: URL.createObjectURL(file), status: "pending", result: null, gapAfter: false });
  }
  renderPhotos();
}

function removePhoto(id) {
  const i = state.photos.findIndex((p) => p.id === id);
  if (i < 0) return;
  URL.revokeObjectURL(state.photos[i].url);
  state.photos.splice(i, 1);
  renderPhotos();
}

function photosToText() {
  return state.photos
    .map((p, i) => (p.result?.label ?? "?").toLowerCase() + (p.gapAfter && i < state.photos.length - 1 ? " " : ""))
    .join("");
}

function landmarkSvg(hands, width, height) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.setAttribute("aria-hidden", "true");
  const r = Math.max(width, height) / 150;
  for (const hand of hands) {
    for (const [a, b] of HAND_CONNECTIONS) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", hand[a][0] * width); line.setAttribute("y1", hand[a][1] * height);
      line.setAttribute("x2", hand[b][0] * width); line.setAttribute("y2", hand[b][1] * height);
      svg.append(line);
    }
    for (const [x, y] of hand) {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", x * width); c.setAttribute("cy", y * height); c.setAttribute("r", r);
      svg.append(c);
    }
  }
  return svg;
}

function renderPhotos() {
  const items = [];
  state.photos.forEach((photo, i) => {
    const li = document.createElement("li");
    li.className = "photo";
    li.dataset.status = photo.status;
    const fig = document.createElement("figure");
    fig.style.margin = "0";
    const frame = document.createElement("div");
    frame.className = "photo-frame";
    const img = document.createElement("img");
    img.src = photo.url;
    img.alt = `Photo ${i + 1}: ${photo.file.name}`;
    frame.append(img);
    if (photo.result?.landmarks?.length) {
      const addOverlay = () => frame.append(landmarkSvg(photo.result.landmarks, img.naturalWidth, img.naturalHeight));
      img.complete && img.naturalWidth ? addOverlay() : img.addEventListener("load", addOverlay, { once: true });
    }
    const cap = document.createElement("figcaption");
    const badge = document.createElement("span");
    badge.className = "badge";
    const conf = document.createElement("span");
    if (photo.status === "done") { badge.textContent = photo.result.label; conf.textContent = pct(photo.result.confidence); }
    else if (photo.status === "nohand") badge.textContent = "No hand found";
    else if (photo.status === "error") { badge.textContent = "Error"; badge.title = photo.error; }
    else if (photo.status === "working") conf.textContent = "Reading…";
    cap.append(badge, conf);
    fig.append(frame, cap);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "remove";
    rm.textContent = "×";
    rm.setAttribute("aria-label", `Remove photo ${i + 1}`);
    rm.onclick = () => removePhoto(photo.id);
    li.append(fig, rm);
    items.push(li);

    if (i < state.photos.length - 1) {
      const gapLi = document.createElement("li");
      const gap = document.createElement("button");
      gap.type = "button";
      gap.className = "gap-toggle";
      gap.setAttribute("aria-pressed", String(photo.gapAfter));
      gap.setAttribute("aria-label", `Space after photo ${i + 1}`);
      gap.textContent = photo.gapAfter ? "space" : "+";
      gap.title = "Toggle a space between these letters";
      gap.onclick = () => { photo.gapAfter = !photo.gapAfter; renderPhotos(); };
      gapLi.append(gap);
      items.push(gapLi);
    }
  });
  el.photoStrip.replaceChildren(...items);
  el.photoActions.hidden = state.photos.length === 0;
  const anyDone = state.photos.some((p) => p.result);
  el.photoText.textContent = anyDone ? photosToText() : "–";
  el.btnPhotosUse.disabled = !anyDone;
}

const PHOTO_MAX_SIDE = 1280;  // large phone photos are scaled down before detection

let imageLandmarker = null;

async function detectPhoto(file) {
  // createImageBitmap applies the EXIF orientation, so sideways-stored phone photos come out upright.
  let bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, PHOTO_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  if (scale < 1) {
    const { width, height } = bitmap;
    bitmap.close();
    bitmap = await createImageBitmap(file, {
      imageOrientation: "from-image", resizeWidth: Math.round(width * scale), resizeHeight: Math.round(height * scale), resizeQuality: "high",
    });
  }
  try {
    const result = imageLandmarker.detect(bitmap);
    return { hands: result.landmarks.map((h) => h.map((p) => [p.x, p.y, p.z])), width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

async function recognizePhotos() {
  if (!state.model) return toast("The model is still loading, try again in a moment");
  setBusy(el.btnPhotosRecognize, true);
  try {
    imageLandmarker ??= await createLandmarker("IMAGE");
  } catch (err) {
    console.error(err);
    setBusy(el.btnPhotosRecognize, false);
    return toast("Could not load the hand tracker. Check your internet connection and try again.", { kind: "error", timeout: 6000 });
  }
  for (const photo of state.photos) {
    if (photo.status === "done" || photo.status === "nohand") continue;
    photo.status = "working";
    renderPhotos();
    try {
      const { hands, width, height } = await detectPhoto(photo.file);
      if (!hands.length) {
        photo.result = { label: null, landmarks: [] };
        photo.status = "nohand";
      } else {
        const pred = state.model.predict(hands, width, height, 3);
        photo.result = { ...pred, landmarks: hands };
        photo.status = "done";
      }
    } catch (err) {
      photo.status = "error";
      photo.error = err.message;
    }
    renderPhotos();
  }
  setBusy(el.btnPhotosRecognize, false);
  if (state.photos.some((p) => p.status === "nohand")) {
    toast("No hand was found in some photos. Try a brighter photo with the whole hand visible.", { timeout: 5000 });
  }
}

el.fileInput.onchange = (e) => { addPhotos(e.target.files); e.target.value = ""; };
for (const evt of ["dragenter", "dragover"]) el.dropzone.addEventListener(evt, (e) => { e.preventDefault(); el.dropzone.classList.add("dragging"); });
for (const evt of ["dragleave", "drop"]) el.dropzone.addEventListener(evt, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragging"); });
el.dropzone.addEventListener("drop", (e) => { if (e.dataTransfer?.files?.length) addPhotos(e.dataTransfer.files); });
el.btnPhotosRecognize.onclick = recognizePhotos;
el.btnPhotosClear.onclick = () => { state.photos.forEach((p) => URL.revokeObjectURL(p.url)); state.photos = []; renderPhotos(); };
el.btnPhotosUse.onclick = () => {
  const text = photosToText();
  if (el.message.value && !el.message.value.endsWith(" ")) typeText(" ");
  typeText(text);
  toast(`Added “${text}” to the message`);
};

// ── Sign guide ──────────────────────────────────────────────────────────────

function renderGuide(classes) {
  el.guideGrid.replaceChildren(...classes.map((c) => {
    const li = document.createElement("li");
    li.dataset.label = c;
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `assets/signs/${encodeURIComponent(c)}.jpg`;
    img.alt = `Sign for ${c}`;
    img.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.textContent = c;
    fig.append(img, cap);
    li.append(fig);
    return li;
  }));
}

function highlightGuide(label) {
  for (const li of el.guideGrid.children) li.classList.toggle("active", li.dataset.label === label);
}

// ── Settings ────────────────────────────────────────────────────────────────

function applySettings() {
  const s = state.settings;
  el.holdTime.value = s.holdMs;
  el.holdTimeOut.textContent = `${(s.holdMs / 1000).toFixed(1)} s`;
  el.minConf.value = s.minConf;
  el.minConfOut.textContent = pct(s.minConf);
  el.autoSpace.checked = s.autoSpace;
}
function saveSettings() {
  state.settings = { holdMs: Number(el.holdTime.value), minConf: Number(el.minConf.value), autoSpace: el.autoSpace.checked };
  storage.set("signspeak.settings", state.settings);
  applySettings();
}
el.holdTime.oninput = saveSettings;
el.minConf.oninput = saveSettings;
el.autoSpace.onchange = saveSettings;

// ── Start ───────────────────────────────────────────────────────────────────

state.settings = { ...state.settings, ...storage.get("signspeak.settings", {}) };
applySettings();
el.btnCameraStart.onclick = startCamera;
el.btnCameraStop.onclick = stopCamera;
renderPrediction(null);
loadModel();
detectServer();

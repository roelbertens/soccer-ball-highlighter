/* =====================================================================
 * Soccer Ball Highlighter — content.js  (v2)
 *
 * Difference from v1:  DETECTOR + TRACKER.
 *   - The detector (YOLOv8, football-trained, class "ball") runs every so
 *     often and finds/confirms the ball. Model: TensorFlow.js GraphModel in
 *     models/<res>/.
 *   - The TRACKER runs EVERY tick and follows the ball between detections
 *     using MOTION + BRIGHTNESS in a small search window, so the ring sticks
 *     to the ball smoothly instead of flickering.
 *   - Manual LOCK: press L and click the ball -> the tracker picks it up.
 *     Always works, even when auto-detection is unsure.
 *
 * Controls:  Alt+B = panel,  Alt+H = on/off,  L = lock mode (then click).
 *
 * Coordinates: "det pixels" = the downscaled detection canvas (detWidth wide).
 *              We only convert to screen pixels when drawing.
 * ===================================================================== */

(() => {
  "use strict";
  if (window.__ballHl2) return;
  window.__ballHl2 = true;

  const S = {
    on: true,
    ringColor: "#00e5ff",
    ringWidth: 5,
    ringScale: 2.0,
    trail: true,
    spotlight: false,
    // detection
    detWidth: 480,          // width of the tracker canvas (small = fast)
    detRes: 800,            // YOLO detection resolution (800/960/1280); higher = more accurate, heavier
    minScore: 0.10,         // YOLO threshold (balls below this score are ignored)
    detectMs: 60,           // minimum time between detections (detector is primary now)
    // tracker (motion/brightness tracker; can drift to bright shirts -> off by default)
    tracker: false,
    tickMs: 30,             // how often the loop runs
    searchRad: 24,          // tracker search radius in det pixels
    trackThresh: 0.12,      // minimum tracker score to stay "locked"
    smooth: 0.4,            // position smoothing (0=instant, 1=very slow)
    holdMs: 450,            // keep showing the last detection this long without a new hit
    coastMs: 180            // how long the ring keeps moving along the last direction (gap bridging)
  };

  let model = null, video = null, overlay = null, octx = null;
  // det = current ball position in DET pixels
  let det = { x: 0, y: 0, r: 6, t: 0, has: false, vx: 0, vy: 0 };
  const trail = [];
  let lastDetect = 0, busyDet = false, warned = false, lockMode = false;

  const detCanvas = document.createElement("canvas");
  const dctx = detCanvas.getContext("2d", { willReadFrequently: true });
  let gray = null, prevGray = null, gw = 0, gh = 0;

  // ---- YOLOv8 model (TensorFlow.js GraphModels in models/<res>/) -----
  const DET_SIZES = [800, 960, 1280];      // available detection resolutions (one model each)
  const BALL_CH = 4;                        // output channel of the "ball" class (4 box + class 0 = ball)
  const yoloCanvas = document.createElement("canvas");
  const yctx = yoloCanvas.getContext("2d", { willReadFrequently: true });
  let coordsNormalized = null;             // pixels vs normalized; determined on 1st detection
  let modelRes = 0;                        // resolution of the CURRENTLY loaded model
  let loadingModel = false;

  // Loads (or switches) the model for a given resolution.
  async function loadModel(res) {
    if (loadingModel || res === modelRes) return;
    loadingModel = true;
    const prev = model;
    try {
      showMsg("⏳ Loading model (" + res + "px)…");
      const m = await tf.loadGraphModel(chrome.runtime.getURL("models/" + res + "/model.json"));
      tf.tidy(() => m.execute(tf.zeros([1, res, res, 3])));   // warm-up (compile shaders)
      model = m; modelRes = res; coordsNormalized = null;
      if (prev && prev !== m) prev.dispose();
      showMsg("✅ Ready (" + res + "px · " + tf.getBackend() + "). Alt+B panel · Alt+H on/off · L = lock on ball.", 4000);
    } catch (e) {
      showMsg("❌ Model load failed: " + e.message);
    } finally { loadingModel = false; }
  }

  // ---- find the video -----------------------------------------------
  function pickVideo() {
    const vids = [...document.querySelectorAll("video")]
      .filter(v => v.videoWidth > 0 && v.clientWidth > 100);
    if (!vids.length) return null;
    return vids.sort((a, b) =>
      b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0];
  }

  // ---- overlay ------------------------------------------------------
  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement("canvas");
    overlay.style.cssText =
      "position:fixed;left:0;top:0;pointer-events:none;z-index:2147483646;";
    (document.fullscreenElement || document.body).appendChild(overlay);
    octx = overlay.getContext("2d");
  }
  function syncOverlay() {
    const parent = document.fullscreenElement || document.body;
    if (overlay.parentElement !== parent) parent.appendChild(overlay);
    const w = window.innerWidth, h = window.innerHeight;
    if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  }
  // where the actual picture content is (accounting for black bars)
  function contentRect() {
    const r = video.getBoundingClientRect();
    const vw = video.videoWidth, vh = video.videoHeight;
    const scale = Math.min(r.width / vw, r.height / vh);
    return { left: r.left + (r.width - vw * scale) / 2,
             top:  r.top  + (r.height - vh * scale) / 2, scale, vw, vh };
  }

  // ---- frame -> grayscale ------------------------------------------
  function grabFrame() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return false;
    gw = S.detWidth; gh = Math.round(gw * vh / vw);
    detCanvas.width = gw; detCanvas.height = gh;
    try { dctx.drawImage(video, 0, 0, gw, gh); }
    catch (e) {
      if (!warned) { warned = true; showMsg("⚠️ Can't read the picture (probably a DRM-protected stream)."); }
      return false;
    }
    const d = dctx.getImageData(0, 0, gw, gh).data;
    if (!gray || gray.length !== gw * gh) { gray = new Float32Array(gw * gh); prevGray = new Float32Array(gw * gh); }
    for (let i = 0, p = 0; i < d.length; i += 4, p++)
      gray[p] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];  // 0..255
    return true;
  }

  // ---- DETECTOR (YOLOv8, heavy, runs occasionally) -------------------
  // Runs the football-trained model on the FULL video frame (the model needs
  // the scene context; zooming into a crop actually lowers confidence).
  // A higher resolution gives the small ball more pixels -> better detection.
  // Coordinates are mapped back: model px -> video px -> det px.
  async function runDetector() {
    if (!model) return null;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return null;
    const R = modelRes;                         // fixed input size of the loaded model
    if (yoloCanvas.width !== R) { yoloCanvas.width = R; yoloCanvas.height = R; }
    const detToV = vw / gw;                     // det pixels -> video pixels

    // letterbox the whole frame (vw x vh) into the R x R square
    const mScale = Math.min(R / vw, R / vh);
    const mPadX = Math.floor((R - Math.round(vw * mScale)) / 2);
    const mPadY = Math.floor((R - Math.round(vh * mScale)) / 2);
    yctx.fillStyle = "#727272";                 // gray (114) like the YOLO letterbox
    yctx.fillRect(0, 0, R, R);
    try { yctx.drawImage(video, 0, 0, vw, vh, mPadX, mPadY, Math.round(vw * mScale), Math.round(vh * mScale)); }
    catch (e) { return null; }                  // protected/DRM stream

    // 2) inference -> [N, C] with C = 4 box (cx,cy,w,h) + class scores
    const t = tf.tidy(() => {
      const img = tf.browser.fromPixels(yoloCanvas).toFloat().div(255).expandDims(0);
      let out = model.execute(img);
      if (Array.isArray(out)) out = out[0];
      out = out.squeeze();                        // [C, N] or [N, C]
      if (out.shape[0] < out.shape[1]) out = out.transpose();  // -> [N, C]
      return out;
    });
    const data = await t.data();
    const [N, C] = t.shape;
    t.dispose();

    // 3) determine the coordinate scale once (pixels 0..R or normalized 0..1)
    if (coordsNormalized === null) {
      let mx = 0;
      for (let i = 0; i < N; i++) mx = Math.max(mx, data[i * C], data[i * C + 1]);
      coordsNormalized = mx <= 1.5;
      console.log("[ball-highlighter] YOLO output", N + "x" + C,
                  coordsNormalized ? "(normalized)" : "(pixels)");
    }
    const unit = coordsNormalized ? R : 1;        // scale box coords to model pixels

    // 4) best candidate: high score + close to the predicted position
    const px = det.has ? det.x + det.vx : gw / 2;
    const py = det.has ? det.y + det.vy : gh / 2;
    let best = null, bestScore = -1;
    for (let i = 0; i < N; i++) {
      const o = i * C;
      const cls = data[o + BALL_CH];              // only the "ball" class (ignore players)
      if (cls < S.minScore) continue;
      // box: model px -> video px (undo letterbox) -> det px
      const cx = ((data[o]     * unit - mPadX) / mScale) / detToV;
      const cy = ((data[o + 1] * unit - mPadY) / mScale) / detToV;
      const r  = (Math.max(data[o + 2], data[o + 3]) * unit / 2 / mScale) / detToV;
      const dist = Math.hypot(cx - px, cy - py);
      const s = cls + Math.max(0, 1 - dist / (gw * 0.5)) * 0.4;
      if (s > bestScore) { bestScore = s; best = { x: cx, y: cy, r }; }
    }
    return best;
  }

  // ---- TRACKER (light, every tick) -----------------------------------
  // Searches a window around the predicted position for the peak of
  // (brightness * motion). The ball is small, bright and moving; field lines
  // are bright but static, so motion separates the ball from them.
  function trackStep() {
    if (!prevGray || !det.has) return false;
    const cx = Math.round(det.x + det.vx);
    const cy = Math.round(det.y + det.vy);
    const R = S.searchRad;
    const x0 = Math.max(1, cx - R), x1 = Math.min(gw - 2, cx + R);
    const y0 = Math.max(1, cy - R), y1 = Math.min(gh - 2, cy + R);
    let bestS = 0, bx = det.x, by = det.y;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const p = y * gw + x;
        // 3x3 sum of (normalized brightness) * (motion)
        let s = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const q = p + dy * gw + dx;
            const bright = gray[q] / 255;
            const motion = Math.abs(gray[q] - prevGray[q]) / 255;
            s += (0.4 + 0.6 * bright) * motion;
          }
        // slight preference for staying close to the prediction (stability)
        const dist = Math.hypot(x - cx, y - cy);
        s *= 1 - Math.min(0.5, dist / (R * 3));
        if (s > bestS) { bestS = s; bx = x; by = y; }
      }
    }
    if (bestS < S.trackThresh) return false;    // ball lost within the window
    // update with smoothing
    const nx = det.x + (bx - det.x) * (1 - S.smooth);
    const ny = det.y + (by - det.y) * (1 - S.smooth);
    det.vx = nx - det.x; det.vy = ny - det.y;
    det.x = nx; det.y = ny; det.t = performance.now(); det.has = true;
    return true;
  }

  function seedFrom(b) {                          // detector/click -> position + velocity
    if (!b) return;
    const now = performance.now();
    if (det.has) {
      const dt = Math.max(16, now - det.t);       // ms since previous hit
      const nvx = (b.x - det.x) / dt, nvy = (b.y - det.y) / dt;  // det px per ms
      det.vx = det.vx * 0.4 + nvx * 0.6;          // EMA smoothing of the velocity
      det.vy = det.vy * 0.4 + nvy * 0.6;
      const vmax = 0.5;                            // cap (det px/ms) against wild jumps
      det.vx = Math.max(-vmax, Math.min(vmax, det.vx));
      det.vy = Math.max(-vmax, Math.min(vmax, det.vy));
    } else { det.vx = 0; det.vy = 0; }
    det.x = b.x; det.y = b.y; if (b.r) det.r = b.r;
    det.t = now; det.has = true;
    trail.push({ x: b.x, y: b.y }); if (trail.length > 16) trail.shift();
  }

  // ---- main loop ------------------------------------------------------
  async function loop() {
    if (video && S.on && grabFrame()) {
      const tracked = S.tracker ? trackStep() : false;
      if (tracked) { trail.push({ x: det.x, y: det.y }); if (trail.length > 16) trail.shift(); }

      const now = performance.now();
      if (model && !busyDet && now - lastDetect >= S.detectMs) {
        busyDet = true; lastDetect = now;
        runDetector().then(b => {
          if (b) seedFrom(b);                     // the detector is in charge: every ball hit places the ring
        }).catch(() => {}).finally(() => { busyDet = false; });
      }
      prevGray.set(gray);
    }
    setTimeout(loop, S.tickMs);
  }

  // ---- drawing --------------------------------------------------------
  function draw(now) {
    requestAnimationFrame(draw);
    if (!video || !overlay) return;
    syncOverlay();
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (lockMode) drawLockHint();
    if (!S.on || !det.has) return;

    const stale = now - det.t;
    if (stale > S.holdMs) { det.has = false; trail.length = 0; return; }
    const alpha = Math.max(0, 1 - stale / S.holdMs);

    // coast: extrapolate along the last ball direction between sparse detections
    // (dead reckoning from the detector velocity; never jumps to bright shirts).
    const coast = S.tracker ? 0 : Math.min(stale, S.coastMs);
    const dx = det.x + det.vx * coast;
    const dy = det.y + det.vy * coast;

    const cr = contentRect();
    const k = cr.vw / gw;                  // det px -> video px
    const sx = cr.left + dx * k * cr.scale;
    const sy = cr.top + dy * k * cr.scale;
    const rr = Math.max(15, det.r * k * cr.scale * S.ringScale);

    if (S.spotlight) {
      octx.save();
      octx.fillStyle = "rgba(0,0,0,0.55)";
      octx.fillRect(0, 0, overlay.width, overlay.height);
      octx.globalCompositeOperation = "destination-out";
      const g = octx.createRadialGradient(sx, sy, rr * 0.6, sx, sy, rr * 1.9);
      g.addColorStop(0, "#000"); g.addColorStop(1, "rgba(0,0,0,0)");
      octx.fillStyle = g; octx.beginPath(); octx.arc(sx, sy, rr * 1.9, 0, 7); octx.fill();
      octx.restore();
    }
    if (S.trail) {
      for (let i = 0; i < trail.length; i++) {
        const p = trail[i];
        octx.beginPath();
        octx.arc(cr.left + p.x * k * cr.scale, cr.top + p.y * k * cr.scale, rr * 0.3, 0, 7);
        octx.fillStyle = S.ringColor; octx.globalAlpha = alpha * 0.10 * (i / trail.length); octx.fill();
      }
      octx.globalAlpha = 1;
    }
    octx.globalAlpha = alpha;
    octx.lineWidth = S.ringWidth + 3; octx.strokeStyle = "rgba(0,0,0,0.85)";
    octx.beginPath(); octx.arc(sx, sy, rr, 0, 7); octx.stroke();
    octx.lineWidth = S.ringWidth; octx.strokeStyle = S.ringColor;
    octx.beginPath(); octx.arc(sx, sy, rr, 0, 7); octx.stroke();
    octx.globalAlpha = 1;
  }
  function drawLockHint() {
    octx.save();
    octx.fillStyle = "#00e5ff"; octx.font = "600 16px system-ui,sans-serif";
    octx.fillText("🎯 Lock mode: click the ball", 20, 34);
    octx.restore();
  }

  // ---- manual lock (L + click) ----------------------------------------
  function enterLock() {
    if (!video) return;
    lockMode = true;
    overlay.style.pointerEvents = "auto";
    overlay.style.cursor = "crosshair";
    showMsg("🎯 Click the ball…");
    overlay.addEventListener("click", onLockClick, { once: true });
  }
  function onLockClick(e) {
    const cr = contentRect();
    const k = cr.vw / gw;
    // screen -> video -> det
    const vx = (e.clientX - cr.left) / cr.scale;
    const vy = (e.clientY - cr.top) / cr.scale;
    seedFrom({ x: vx / k, y: vy / k, r: det.r || 6 });
    lockMode = false;
    overlay.style.pointerEvents = "none";
    overlay.style.cursor = "";
    showMsg("✅ Locked on the ball.", 2000);
  }

  // ---- startup --------------------------------------------------------
  async function boot() {
    video = pickVideo();
    if (!video) { setTimeout(boot, 1500); return; }
    ensureOverlay();
    try {
      try { await tf.setBackend("webgl"); await tf.ready(); }
      catch (glErr) { await tf.setBackend("cpu"); await tf.ready(); }
    } catch (e) { showMsg("❌ TensorFlow failed to start: " + e.message); return; }
    await loadModel(S.detRes);                    // load the chosen model + warm-up
    requestAnimationFrame(draw);
    loop();
  }
  document.addEventListener("fullscreenchange", () => {
    if (overlay) (document.fullscreenElement || document.body).appendChild(overlay);
  });

  // ---- panel + keys ---------------------------------------------------
  let panel, msgEl;
  function showMsg(t, ms) {
    if (!msgEl) return console.log("[ball-highlighter]", t);
    msgEl.textContent = t;
    if (ms) setTimeout(() => { if (msgEl.textContent === t) msgEl.textContent = ""; }, ms);
  }
  function buildPanel() {
    panel = document.createElement("div");
    panel.style.cssText =
      "position:fixed;right:16px;bottom:16px;z-index:2147483647;display:none;" +
      "font:13px system-ui,sans-serif;color:#fff;background:#111d;backdrop-filter:blur(6px);" +
      "border:1px solid #00e5ff55;border-radius:10px;padding:12px 14px;width:246px;box-shadow:0 6px 24px #0009;";
    panel.innerHTML = `
      <div style="font-weight:600;margin-bottom:8px;color:#00e5ff">⚽ Ball Highlighter v2</div>
      <label style="display:flex;justify-content:space-between;margin:5px 0">Highlight<input type="checkbox" id="bh-on" checked></label>
      <label style="display:flex;justify-content:space-between;margin:5px 0">Tracker (smooth, may drift)<input type="checkbox" id="bh-trk"></label>
      <label style="display:flex;justify-content:space-between;margin:5px 0">Trail<input type="checkbox" id="bh-trail" checked></label>
      <label style="display:flex;justify-content:space-between;margin:5px 0">Spotlight (dim the rest)<input type="checkbox" id="bh-spot"></label>
      <label style="display:block;margin:8px 0 2px">Ring size</label>
      <input type="range" id="bh-size" min="1" max="4" step="0.1" value="2.0" style="width:100%">
      <label style="display:block;margin:8px 0 2px">Detection sensitivity</label>
      <input type="range" id="bh-score" min="0.03" max="0.6" step="0.01" value="0.10" style="width:100%">
      <label style="display:block;margin:8px 0 2px">Tracker grip</label>
      <input type="range" id="bh-trh" min="0.03" max="0.4" step="0.01" value="0.12" style="width:100%">
      <label style="display:block;margin:8px 0 2px">Detection resolution (higher = more accurate, heavier)</label>
      <div id="bh-res" style="display:flex;gap:6px;margin-top:4px"></div>
      <label style="display:block;margin:8px 0 2px">Ring color</label>
      <div id="bh-colors" style="display:flex;gap:6px;margin-top:4px"></div>
      <button id="bh-lock" style="margin-top:10px;width:100%;padding:6px;border:0;border-radius:6px;background:#00e5ff;color:#003;font-weight:600;cursor:pointer">🎯 Lock on ball (or press L)</button>
      <div id="bh-msg" style="margin-top:8px;min-height:16px;color:#9fe;font-size:12px"></div>`;
    document.body.appendChild(panel);
    msgEl = panel.querySelector("#bh-msg");
    const $ = id => panel.querySelector(id);
    $("#bh-on").onchange    = e => S.on = e.target.checked;
    $("#bh-trk").onchange   = e => S.tracker = e.target.checked;
    $("#bh-trail").onchange = e => S.trail = e.target.checked;
    $("#bh-spot").onchange  = e => S.spotlight = e.target.checked;
    $("#bh-size").oninput   = e => S.ringScale = +e.target.value;
    $("#bh-score").oninput  = e => S.minScore = +e.target.value;
    $("#bh-trh").oninput    = e => S.trackThresh = +e.target.value;
    $("#bh-lock").onclick   = enterLock;
    ["#00e5ff","#ffe500","#ff2bd6","#00ff6a","#ffffff"].forEach(c => {
      const b = document.createElement("button");
      b.style.cssText = `width:30px;height:22px;border-radius:5px;border:1px solid #fff5;background:${c};cursor:pointer`;
      b.onclick = () => S.ringColor = c;
      $("#bh-colors").appendChild(b);
    });
    // resolution picker: one button per available model, active one highlighted
    const resWrap = $("#bh-res");
    function renderRes() {
      resWrap.innerHTML = "";
      DET_SIZES.forEach(sz => {
        const b = document.createElement("button");
        const active = sz === S.detRes;
        b.textContent = sz + "px";
        b.style.cssText = "flex:1;padding:5px 0;border-radius:5px;cursor:pointer;font:600 12px system-ui;" +
          (active ? "background:#00e5ff;color:#003;border:0" : "background:transparent;color:#9fe;border:1px solid #00e5ff55");
        b.onclick = () => { if (sz === S.detRes) return; S.detRes = sz; renderRes(); loadModel(sz); };
        resWrap.appendChild(b);
      });
    }
    renderRes();
  }
  window.addEventListener("keydown", e => {
    if (e.code === "KeyL" && !e.altKey && !/input|textarea/i.test(e.target.tagName)) { enterLock(); return; }
    if (!e.altKey) return;
    if (e.code === "KeyB") { if (!panel) buildPanel(); panel.style.display = panel.style.display === "none" ? "block" : "none"; }
    if (e.code === "KeyH") { S.on = !S.on; if (panel) panel.querySelector("#bh-on").checked = S.on; }
  });

  buildPanel();
  boot();
})();

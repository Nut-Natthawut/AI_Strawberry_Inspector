"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as ort from "onnxruntime-web";

// 3 classes: Ripe, Turning, Unripe (trained directly)

// Color & emoji mapping for 3 display classes
const CLASS_STYLES: Record<string, { color: string; emoji: string; bg: string; border: string; barColor: string; boxColor: string }> = {
  "Ripe": { color: "text-red-400", emoji: "🍓", bg: "bg-red-500/20", border: "border-red-500/50", barColor: "#f87171", boxColor: "#f87171" },
  "Turning": { color: "text-amber-400", emoji: "🟡", bg: "bg-amber-500/20", border: "border-amber-500/50", barColor: "#fbbf24", boxColor: "#fbbf24" },
  "Unripe": { color: "text-green-400", emoji: "🟢", bg: "bg-green-500/20", border: "border-green-500/50", barColor: "#4ade80", boxColor: "#4ade80" },
};

const DEFAULT_STYLE = { color: "text-white", emoji: "❓", bg: "bg-slate-700/50", border: "border-slate-600", barColor: "#94a3b8", boxColor: "#06b6d4" };

const getStrawberryStyle = (className: string) => CLASS_STYLES[className] || DEFAULT_STYLE;
const getBoxColor = (className: string) => (CLASS_STYLES[className]?.boxColor) || "#06b6d4";

interface Detection {
  x1: number; y1: number; x2: number; y2: number;
  classId: number; className: string; score: number;
}

// --- Tracking Structure ---
interface Track {
  id: number;
  classId: number;
  className: string;
  cx: number;
  cy: number;
  lastSeenTime: number; // timestamp
  framesVisible: number;
  color: string;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null); // transparent overlay for boxes
  const offscreenRef = useRef<HTMLCanvasElement | null>(null); // hidden canvas for pixel capture

  const [status, setStatus] = useState<string>("Initializing system...");
  const [topDetection, setTopDetection] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({ Ripe: 0, Turning: 0, Unripe: 0 });
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [showSummary, setShowSummary] = useState<boolean>(false);

  // Tracking unique strawberries seen over the entire session
  const [sessionStats, setSessionStats] = useState({
    Ripe: 0,
    Turning: 0,
    Unripe: 0,
    framesAnalyzed: 0,
    totalUniqueDetected: 0,
    totalRipe: 0,
    totalTurning: 0,
    totalUnripe: 0
  });

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const inferenceRunningRef = useRef<boolean>(false);
  const detectionsRef = useRef<Detection[]>([]);
  const activeTracksRef = useRef<Track[]>([]);
  const trackIdCounterRef = useRef<number>(1);
  const totalValidTracksRef = useRef<number>(0);

  // Accumulated unique totals by class
  const totalRipeRef = useRef<number>(0);
  const totalTurningRef = useRef<number>(0);
  const totalUnripeRef = useRef<number>(0);
  const animFrameRef = useRef<number>(0);
  const inferenceTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- Load ONNX Model ---
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/best_3class.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("Failed to load classes.json");
    classesRef.current = await clsRes.json();
  }

  // --- Preprocess: capture pixels from offscreen canvas → YOLO tensor ---
  function preprocessToTensor(video: HTMLVideoElement) {
    const size = 640;
    let offscreen = offscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      offscreenRef.current = offscreen;
    }

    // Draw video into a small offscreen canvas
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    offscreen.width = size;
    offscreen.height = size;
    const ctx = offscreen.getContext("2d", { willReadFrequently: true })!;

    const scale = Math.min(size / vw, size / vh);
    const sw = Math.floor(vw * scale);
    const sh = Math.floor(vh * scale);
    const dx = Math.floor((size - sw) / 2);
    const dy = Math.floor((size - sh) / 2);

    ctx.fillStyle = "#808080";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(video, dx, dy, sw, sh);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(3 * size * size);
    const tp = size * size;

    for (let i = 0; i < tp; i++) {
      const si = i * 4;
      float[i] = imgData[si] / 255;
      float[i + tp] = imgData[si + 1] / 255;
      float[i + tp * 2] = imgData[si + 2] / 255;
    }

    return {
      tensor: new ort.Tensor("float32", float, [1, 3, size, size]),
      scale, dx, dy,
    };
  }

  // --- YOLO Decode [1, 10, 8400] column-major ---
  function decodeYOLO(
    output: ort.Tensor,
    classes: string[],
    lb: { scale: number; dx: number; dy: number }
  ): Detection[] {
    const data = output.data as Float32Array;
    const numBoxes = output.dims[2];
    const numClasses = classes.length;
    const confThreshold = 0.4; // ⬆️ เพิ่มให้จับเฉพาะลูกที่มั่นใจ ลดมั่ว
    const iouThreshold = 0.25; // ⬇️ ลดให้ซ้อนทับกันได้น้อยลง เคลียร์ box ซ้ำ
    const boxes: Detection[] = [];

    for (let i = 0; i < numBoxes; i++) {
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

      let maxScore = -Infinity;
      let classId = 0;
      for (let c = 0; c < numClasses; c++) {
        const s = data[(4 + c) * numBoxes + i];
        if (s > maxScore) { maxScore = s; classId = c; }
      }
      if (maxScore < confThreshold) continue;

      const displayName = classes[classId] || "Unknown";

      boxes.push({
        x1: (cx - w / 2 - lb.dx) / lb.scale,
        y1: (cy - h / 2 - lb.dy) / lb.scale,
        x2: (cx + w / 2 - lb.dx) / lb.scale,
        y2: (cy + h / 2 - lb.dy) / lb.scale,
        classId, className: displayName, score: maxScore,
      });
    }

    // NMS
    boxes.sort((a, b) => b.score - a.score);
    const sel: Detection[] = [];
    while (boxes.length > 0) {
      const best = boxes.shift()!;
      sel.push(best);
      for (let i = boxes.length - 1; i >= 0; i--) {
        if (computeIoU(best, boxes[i]) > iouThreshold) boxes.splice(i, 1);
      }
    }
    return sel;
  }

  function computeIoU(a: Detection, b: Detection) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const aB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aA + aB - inter + 1e-6);
  }

  // --- Run inference (called by setInterval, NOT in render loop) ---
  async function runInference() {
    if (inferenceRunningRef.current) return;
    const session = sessionRef.current;
    const classes = classesRef.current;
    const video = videoRef.current;
    if (!session || !classes || !video || video.videoWidth === 0) return;

    inferenceRunningRef.current = true;

    try {
      const { tensor, scale, dx, dy } = preprocessToTensor(video);
      const feeds: Record<string, ort.Tensor> = {};
      feeds[session.inputNames[0]] = tensor;

      const out = await session.run(feeds);
      const detections = decodeYOLO(out[session.outputNames[0]], classes, { scale, dx, dy });

      detectionsRef.current = detections;

      // --- Object Tracking Logic ---
      const currentTime = Date.now();
      const currentTracks = activeTracksRef.current;
      const MAX_DISTANCE = 0.40; // Normalized maximum allowed distance to link a detection to an old track
      const TIMEOUT_MS = 3000;   // How long to remember a track if we stop seeing it

      // 1. Calculate centroids for new detections in normalized screen space (0-1)
      const newCenters = detections.map(d => ({
        ...d,
        cx: ((d.x1 + d.x2) / 2) / video.videoWidth,
        cy: ((d.y1 + d.y2) / 2) / video.videoHeight
      }));

      // 2. Match existing tracks to new detections based on shortest distance
      const matchedDetectionIndices = new Set<number>();
      for (const track of currentTracks) {
        let bestDist = Infinity;
        let bestIdx = -1;

        for (let i = 0; i < newCenters.length; i++) {
          if (matchedDetectionIndices.has(i)) continue;

          const d = newCenters[i];
          // Only match if it's the same class type!
          if (d.classId !== track.classId) continue;

          // Euclidean distance between normalized centroids (0-1 space)
          const dist = Math.sqrt(Math.pow(d.cx - track.cx, 2) + Math.pow(d.cy - track.cy, 2));
          if (dist < bestDist && dist < MAX_DISTANCE) {
            bestDist = dist;
            bestIdx = i;
          }
        }

        if (bestIdx !== -1) {
          // Found a match: Update track position and timestamp
          const d = newCenters[bestIdx];
          track.cx = d.cx;
          track.cy = d.cy;
          track.lastSeenTime = currentTime;

          if (track.framesVisible === 0) {
            // First time it is MATCHED (now seen in 2 consecutive inferences minimum)
            // Graduating from "ghost" 1-frame track to "valid" track
            totalValidTracksRef.current++;
            if (track.className === "Ripe") totalRipeRef.current++;
            else if (track.className === "Turning") totalTurningRef.current++;
            else if (track.className === "Unripe") totalUnripeRef.current++;
          }
          track.framesVisible += 1;

          matchedDetectionIndices.add(bestIdx);
        }
      }

      // 3. Create new tracks for unmatched detections
      for (let i = 0; i < newCenters.length; i++) {
        if (!matchedDetectionIndices.has(i)) {
          const d = newCenters[i];
          currentTracks.push({
            id: trackIdCounterRef.current++,
            classId: d.classId,
            className: d.className,
            cx: d.cx,
            cy: d.cy,
            lastSeenTime: currentTime,
            framesVisible: 0, // Starts at 0, must be matched at least once to become a "real" track
            color: getBoxColor(d.className)
          });
        }
      }

      // 4. Remove expired tracks (strawberries that left the frame or were covered)
      activeTracksRef.current = currentTracks.filter(t => (currentTime - t.lastSeenTime) < TIMEOUT_MS);
      // Because inference is only 1 FPS, waiting >1 frame means a 2-second delay. We set it to 1 to feel responsive.
      const activeStableTracks = activeTracksRef.current.filter(t => t.framesVisible >= 1);

      // Update UI counts based on ACTIVE STABLE TRACKS instead of raw detections
      if (activeStableTracks.length > 0) {
        // Find highest confidence detection out of current raw detections for the main UI display
        const topRaw = detections.length > 0 ? detections.reduce((a, b) => (a.score > b.score ? a : b)) : null;
        if (topRaw) {
          setTopDetection(topRaw.className);
          setConf(topRaw.score);
        }

        setDetectionCount(activeStableTracks.length);

        // Count ACTIVE tracks per class
        const counts: Record<string, number> = { Ripe: 0, Turning: 0, Unripe: 0 };
        activeStableTracks.forEach(t => { counts[t.className] = (counts[t.className] || 0) + 1; });
        setClassCounts(counts);

        // Update Session Stats 
        // We track the *MAXIMUM ACTIVE* counted at any single moment, combining with historical memory
        // But for totals, we use the strictly accumulated unique values
        setSessionStats(prev => ({
          ...prev,
          Ripe: counts.Ripe,
          Turning: counts.Turning,
          Unripe: counts.Unripe,
          framesAnalyzed: prev.framesAnalyzed + 1,
          totalUniqueDetected: totalValidTracksRef.current,
          totalRipe: totalRipeRef.current,
          totalTurning: totalTurningRef.current,
          totalUnripe: totalUnripeRef.current
        }));
      } else {
        setTopDetection("-");
        setConf(0);
        setDetectionCount(0);
        setClassCounts({ Ripe: 0, Turning: 0, Unripe: 0 });
        setSessionStats(prev => ({ ...prev, framesAnalyzed: prev.framesAnalyzed + 1 }));
      }
    } catch (e) {
      console.error("Inference error:", e);
    }

    inferenceRunningRef.current = false;
  }

  // --- Overlay render loop (only draws bounding boxes AND tracking IDs) ---
  const renderOverlay = useCallback(function drawOverlay() {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) {
      animFrameRef.current = requestAnimationFrame(drawOverlay);
      return;
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) {
      animFrameRef.current = requestAnimationFrame(drawOverlay);
      return;
    }

    if (overlay.width !== vw || overlay.height !== vh) {
      overlay.width = vw;
      overlay.height = vh;
    }

    const ctx = overlay.getContext("2d")!;
    ctx.clearRect(0, 0, vw, vh);

    // Draw bounding boxes from last inference result + tracking IDs
    // We match bounding boxes to tracking IDs the same way for display purposes. Best effort.
    const dets = detectionsRef.current;
    const activeTracks = activeTracksRef.current;

    for (const det of dets) {
      const color = getBoxColor(det.className);
      const bx = det.x1, by = det.y1, bw = det.x2 - det.x1, bh = det.y2 - det.y1;
      const bcx = ((det.x1 + det.x2) / 2) / vw;
      const bcy = ((det.y1 + det.y2) / 2) / vh;

      // Attempt to find corresponding track ID just for display
      let trackIdStr = "";
      let minD = Infinity;
      for (const t of activeTracks) {
        if (t.classId !== det.classId) continue;
        const d = Math.sqrt(Math.pow(bcx - t.cx, 2) + Math.pow(bcy - t.cy, 2));
        if (d < minD && d < 0.3) {
          minD = d;
          trackIdStr = `#${t.id}`;
        }
      }

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(bx, by, bw, bh);

      // Label (include Track ID)
      const text = `${det.className} ${trackIdStr} ${(det.score * 100).toFixed(0)}%`;
      ctx.font = "bold 14px Inter, sans-serif";
      const tw = ctx.measureText(text).width;
      const pad = 6;
      const lh = 22;
      const ly = by - lh > 0 ? by - lh : by;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, ly, tw + pad * 2, lh, 4);
      ctx.fill();
      ctx.fillStyle = "#000";
      ctx.fillText(text, bx + pad, ly + 16);
    }

    animFrameRef.current = requestAnimationFrame(drawOverlay);
  }, []);

  // --- Start Camera ---
  async function startCamera() {
    setStatus("Requesting camera access...");

    // Reset trackers
    activeTracksRef.current = [];
    trackIdCounterRef.current = 1;
    totalValidTracksRef.current = 0;
    totalRipeRef.current = 0;
    totalTurningRef.current = 0;
    totalUnripeRef.current = 0;
    setSessionStats({ Ripe: 0, Turning: 0, Unripe: 0, framesAnalyzed: 0, totalUniqueDetected: 0, totalRipe: 0, totalTurning: 0, totalUnripe: 0 });
    setShowSummary(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("System Active");
      setIsStreaming(true);

      // Start overlay render loop (lightweight, just bounding boxes)
      animFrameRef.current = requestAnimationFrame(renderOverlay);

      // Start inference on a separate timer (every 1 second)
      inferenceTimerRef.current = setInterval(() => {
        runInference();
      }, 1000);
    } catch {
      setStatus("Camera access denied");
    }
  }

  // --- Stop Camera & Show Summary ---
  function stopCamera() {
    if (inferenceTimerRef.current) {
      clearInterval(inferenceTimerRef.current);
      inferenceTimerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }

    // Stop video tracks
    const video = videoRef.current;
    if (video && video.srcObject) {
      const stream = video.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }

    // Clear canvas
    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    setIsStreaming(false);
    setStatus("Inspection Completed");
    setTopDetection("-");
    setConf(0);
    setDetectionCount(0);
    setClassCounts({ Ripe: 0, Turning: 0, Unripe: 0 });
    setShowSummary(true);
  }

  // --- Boot ---
  useEffect(() => {
    (async () => {
      try {
        setStatus("Loading YOLO Model...");
        await loadModel();
        setStatus("Ready to Start");
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus(`Initialization Failed: ${msg}`);
      }
    })();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (inferenceTimerRef.current) clearInterval(inferenceTimerRef.current);
    };
  }, []);

  const strawberryStyle = getStrawberryStyle(topDetection);

  return (
    <main className="min-h-screen bg-[#0f172a] text-slate-100 font-sans selection:bg-cyan-500 selection:text-white overflow-hidden">
      {/* Background Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-600/20 rounded-full blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-10 flex flex-col items-center gap-8">

        {/* Full Screen Summary Modal */}
        {showSummary && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-xl" />
            <div className="relative w-full max-w-lg bg-slate-900 border border-slate-700/50 rounded-3xl overflow-hidden shadow-2xl shadow-cyan-900/20 flex flex-col items-center p-8 animate-in fade-in zoom-in-95 duration-300">
              <div className="w-16 h-16 rounded-full bg-cyan-500/20 border border-cyan-500/50 flex items-center justify-center mb-6">
                <span className="text-3xl">📋</span>
              </div>

              <h2 className="text-3xl font-extrabold text-white mb-2 text-center">Inspection Summary</h2>
              <p className="text-slate-400 text-center mb-6">
                Analyzed <span className="text-white font-medium">{sessionStats.framesAnalyzed}</span> frames. <br />
                Detected <span className="text-cyan-400 font-bold text-lg">{sessionStats.totalUniqueDetected}</span> individual strawberries.
              </p>

              <div className="w-full space-y-4 mb-8">
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider text-center">Total Breakdown by Ripeness</h3>

                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex flex-col items-center shadow-inner shadow-red-500/5">
                    <span className="text-2xl mb-1">🍓</span>
                    <span className="text-2xl font-bold text-red-400">{sessionStats.totalRipe}</span>
                    <span className="text-xs text-red-400/80 mt-1 uppercase tracking-wider font-semibold">Ripe</span>
                  </div>
                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex flex-col items-center shadow-inner shadow-amber-500/5">
                    <span className="text-2xl mb-1">🟡</span>
                    <span className="text-2xl font-bold text-amber-400">{sessionStats.totalTurning}</span>
                    <span className="text-xs text-amber-400/80 mt-1 uppercase tracking-wider font-semibold">Turning</span>
                  </div>
                  <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 flex flex-col items-center shadow-inner shadow-green-500/5">
                    <span className="text-2xl mb-1">🟢</span>
                    <span className="text-2xl font-bold text-green-400">{sessionStats.totalUnripe}</span>
                    <span className="text-xs text-green-400/80 mt-1 uppercase tracking-wider font-semibold">Unripe</span>
                  </div>
                </div>
              </div>

              {/* AI Assessment */}
              <div className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 mb-8">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-cyan-400">✨</span>
                  <h4 className="font-semibold text-cyan-400">AI Assessment</h4>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed">
                  {sessionStats.totalUniqueDetected === 0 ? "ไม่พบสตรอว์เบอร์รีในระหว่างการตรวจสอบครั้งนี้"
                    : sessionStats.totalRipe > Math.max(sessionStats.totalTurning, sessionStats.totalUnripe) ? "ผลผลิตส่วนใหญ่สุกและพร้อมเก็บเกี่ยวแล้ว โปรดจัดเก็บอย่างระมัดระวังเพื่อป้องกันรอยช้ำ"
                      : sessionStats.totalTurning > sessionStats.totalRipe ? "ผลผลิตกำลังเปลี่ยนสี (เริ่มสุก) ช่วงเวลาเก็บเกี่ยวที่เหมาะสมที่สุดกำลังจะมาถึง แนะนำให้รอประมาณ 1-2 วัน"
                        : "ผลผลิตส่วนใหญ่ยังดิบอยู่ จำเป็นต้องใช้ระยะเวลาในการเจริญเติบโตบนต้นเพิ่มเติม"}
                </p>
              </div>

              <button
                onClick={() => setShowSummary(false)}
                className="w-full py-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-bold border border-slate-600 transition-colors"
              >
                Close & Start Over
              </button>
            </div>
          </div>
        )}

        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-pink-400 to-green-400">
            🍓 AI Strawberry Inspector
          </h1>
          <p className="text-slate-400 text-sm md:text-base">
            Real-time Strawberry Quality Detection using{" "}
            <span className="text-cyan-400 font-medium">YOLOv11</span> &{" "}
            <span className="text-purple-400 font-medium">ONNX Runtime</span>
          </p>
        </div>

        {/* Main Grid */}
        <div className="w-full grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          {/* Camera Feed — video shows directly, canvas overlays bounding boxes */}
          <div className="lg:col-span-2 w-full aspect-video bg-slate-800/50 rounded-2xl border border-slate-700/50 shadow-2xl shadow-cyan-900/20 overflow-hidden relative backdrop-blur-sm">
            {/* Video plays directly = always smooth */}
            <video
              ref={videoRef}
              className={`w-full h-full object-cover ${isStreaming ? "block" : "hidden"}`}
              playsInline
              muted
            />
            {/* Transparent canvas overlay for bounding boxes */}
            <canvas
              ref={overlayRef}
              className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
              style={{ display: isStreaming ? "block" : "none" }}
            />

            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 space-y-4">
                <div className="w-16 h-16 rounded-full border-2 border-slate-600 border-dashed animate-pulse flex items-center justify-center">
                  <span className="text-3xl">🍓</span>
                </div>
                <p>Point your camera at strawberries to inspect them</p>
              </div>
            )}

            {/* Status */}
            <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-xs font-medium border border-white/10 flex items-center gap-2 z-20">
              <span className={`w-2 h-2 rounded-full ${isStreaming ? "bg-green-500 animate-pulse" : "bg-yellow-500"}`} />
              {status}
            </div>

            {isStreaming && detectionCount > 0 && (
              <div className="absolute top-4 right-4 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-xs font-medium border border-white/10 z-20">
                {detectionCount} detected
              </div>
            )}
          </div>

          {/* Right Column */}
          <div className="w-full flex flex-col gap-6">
            {/* Result Card */}
            <div className={`p-6 rounded-2xl border backdrop-blur-md transition-all duration-300 ${strawberryStyle.bg} ${strawberryStyle.border}`}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold uppercase tracking-wider opacity-70">Top Detection</h3>
                <span className="text-3xl">{strawberryStyle.emoji}</span>
              </div>
              <h2 className={`text-4xl font-bold ${strawberryStyle.color} capitalize`}>
                {topDetection === "-" ? "Waiting..." : topDetection}
              </h2>
              <div className="mt-6 space-y-2">
                <div className="flex justify-between text-xs font-medium opacity-80">
                  <span>Confidence</span>
                  <span>{(conf * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-900/20 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${conf * 100}%`, backgroundColor: topDetection === "-" ? "#64748b" : strawberryStyle.barColor }}
                  />
                </div>
              </div>
            </div>

            {/* Strawberry Count */}
            <div className="p-4 rounded-2xl border border-slate-700/50 bg-slate-800/30 backdrop-blur-md relative overflow-hidden">
              <div className="absolute top-0 right-0 p-3 opacity-20">
                {isStreaming && <span className="text-xs font-mono uppercase tracking-widest text-cyan-500 font-bold tracking-indicator">Tracking Active</span>}
              </div>

              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Currently Tracked</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-400" /> Ripe 🍓</div>
                  <span className="text-xl font-bold text-red-400">{classCounts.Ripe}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-amber-400" /> Turning 🟡</div>
                  <span className="text-xl font-bold text-amber-400">{classCounts.Turning}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-green-400" /> Unripe 🟢</div>
                  <span className="text-xl font-bold text-green-400">{classCounts.Unripe}</span>
                </div>
                <div className="border-t border-slate-700/50 pt-2 flex items-center justify-between">
                  <span className="text-sm text-slate-400 font-medium">Total</span>
                  <span className="text-xl font-bold text-white">{detectionCount}</span>
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="p-6 rounded-2xl border border-slate-700/50 bg-slate-800/30 backdrop-blur-md space-y-4">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Control Center</h3>
              {!isStreaming ? (
                <button
                  onClick={startCamera}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-400 hover:to-pink-500 text-white font-bold shadow-lg shadow-red-500/25 transition-all active:scale-95 flex items-center justify-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Start Strawberry Detection
                </button>
              ) : (
                <button
                  onClick={stopCamera}
                  className="w-full py-4 rounded-xl relative group overflow-hidden bg-slate-800 border border-red-500/30 hover:border-red-500/60 shadow-lg shadow-red-900/10 transition-all active:scale-95 flex items-center justify-center gap-2"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-red-600/10 to-orange-600/10 group-hover:from-red-600/20 group-hover:to-orange-600/20 transition-all" />
                  <div className="w-3 h-3 rounded-sm bg-red-500 relative z-10 animate-pulse" />
                  <span className="text-red-400 font-bold relative z-10 group-hover:text-red-300 transition-colors">Stop Inspection & View Summary</span>
                </button>
              )}
              <p className="text-xs text-slate-500 text-center leading-relaxed">
                All processing runs locally in your browser via WebAssembly. No data is sent to any server.
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
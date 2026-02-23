// Web Worker for ONNX inference (runs on separate thread)
import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
let classes: string[] | null = null;

interface Box {
  x1: number; y1: number; x2: number; y2: number;
  classId: number; className: string; score: number;
}

interface Letterbox {
  scale: number; dx: number; dy: number;
}

// Initialize model
async function initModel() {
  try {
    session = await ort.InferenceSession.create(
      "/models/strawberry.onnx",
      { executionProviders: ["wasm"] }
    );

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("Failed to load classes.json");
    classes = await clsRes.json();

    self.postMessage({ type: "status", status: "Ready to Start" });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    self.postMessage({ type: "status", status: `Initialization Failed: ${msg}` });
  }
}

// Decode YOLO output [1, 10, 8400]
function decodeYOLO(
  data: Float32Array,
  dims: readonly number[],
  cls: string[],
  letterbox: Letterbox,
  canvasWidth: number,
  canvasHeight: number
): Box[] {
  const numBoxes = dims[2];
  const numClasses = cls.length;
  const confThreshold = 0.25;
  const iouThreshold = 0.45;
  const boxes: Box[] = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    let maxScore = -Infinity;
    let classId = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore < confThreshold) continue;

    const x1 = (cx - w / 2 - letterbox.dx) / letterbox.scale;
    const y1 = (cy - h / 2 - letterbox.dy) / letterbox.scale;
    const x2 = (cx + w / 2 - letterbox.dx) / letterbox.scale;
    const y2 = (cy + h / 2 - letterbox.dy) / letterbox.scale;

    boxes.push({ x1, y1, x2, y2, classId, className: cls[classId], score: maxScore });
  }

  // NMS
  boxes.sort((a, b) => b.score - a.score);
  const selected: Box[] = [];
  while (boxes.length > 0) {
    const best = boxes.shift()!;
    selected.push(best);
    for (let i = boxes.length - 1; i >= 0; i--) {
      if (computeIoU(best, boxes[i]) > iouThreshold) {
        boxes.splice(i, 1);
      }
    }
  }
  return selected;
}

function computeIoU(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

// Handle messages from main thread
self.onmessage = async (e: MessageEvent) => {
  const { type } = e.data;

  if (type === "init") {
    self.postMessage({ type: "status", status: "Loading YOLO Model..." });
    await initModel();
    return;
  }

  if (type === "infer") {
    if (!session || !classes) return;

    const { imageData, width, height, canvasWidth, canvasHeight } = e.data;
    const pixels = new Uint8ClampedArray(imageData);

    // Preprocess: letterbox to 640x640
    const size = 640;
    const scale = Math.min(size / width, size / height);
    const scaledWidth = Math.floor(width * scale);
    const scaledHeight = Math.floor(height * scale);
    const dx = Math.floor((size - scaledWidth) / 2);
    const dy = Math.floor((size - scaledHeight) / 2);

    const float = new Float32Array(3 * size * size);
    const totalPixels = size * size;

    // Fill with gray (0.5)
    float.fill(0.5);

    // Copy scaled image pixels into tensor
    for (let sy = 0; sy < scaledHeight; sy++) {
      for (let sx = 0; sx < scaledWidth; sx++) {
        const origX = Math.min(Math.floor(sx / scale), width - 1);
        const origY = Math.min(Math.floor(sy / scale), height - 1);

        const srcIdx = (origY * width + origX) * 4;
        const dstX = sx + dx;
        const dstY = sy + dy;
        const dstIdx = dstY * size + dstX;

        float[dstIdx] = pixels[srcIdx] / 255;
        float[dstIdx + totalPixels] = pixels[srcIdx + 1] / 255;
        float[dstIdx + totalPixels * 2] = pixels[srcIdx + 2] / 255;
      }
    }

    try {
      const input = new ort.Tensor("float32", float, [1, 3, size, size]);
      const feeds: Record<string, ort.Tensor> = {};
      feeds[session.inputNames[0]] = input;

      const out = await session.run(feeds);
      const outName = session.outputNames[0];
      const result = out[outName];
      const outputData = result.data as Float32Array;
      const outputDims = result.dims;

      const detections = decodeYOLO(
        outputData, outputDims, classes,
        { scale, dx, dy },
        canvasWidth, canvasHeight
      );

      self.postMessage({ type: "result", detections });
    } catch (err) {
      console.error("Worker inference error:", err);
      self.postMessage({ type: "result", detections: [] });
    }
  }
};

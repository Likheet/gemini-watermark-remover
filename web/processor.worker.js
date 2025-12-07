// Image Processor Web Worker
// Use ort.wasm.min.js - WASM-only build that works with importScripts (no ES modules needed)
importScripts('lib/ort.wasm.min.js');

let modelSession = null;
let isModelLoaded = false;
let modelInputName = null;
let modelOutputName = null;

// PERFORMANCE: Cache star templates to avoid recreation for each image
const templateCache = new Map();

// Suppress "Unknown CPU vendor" logs
const originalWarn = console.warn;
const originalLog = console.log;
const filterFn = (args) => {
    const msg = args[0];
    return typeof msg === 'string' && (msg.includes('Unknown CPU vendor') || msg.includes('cpuinfo_vendor value'));
};
console.warn = function (...args) { if (!filterFn(args)) originalWarn.apply(console, args); };
console.log = function (...args) { if (!filterFn(args)) originalLog.apply(console, args); };

// Configure ONNX Runtime BEFORE any initialization for MAXIMUM PERFORMANCE
ort.env.logLevel = 'error';

// PERFORMANCE: Enable SIMD for ~4x faster AI inference
// SIMD uses CPU vector instructions to process multiple data points simultaneously
ort.env.wasm.simd = true;

// Keep single-threaded to avoid lag spikes on the main thread
// Multi-threading can cause resource contention on lower-end devices
ort.env.wasm.numThreads = 1;

// Disable proxy to avoid overhead
ort.env.wasm.proxy = false;

// Set wasm paths early
ort.env.wasm.wasmPaths = 'lib/';

self.onmessage = async (e) => {
    const { type, data, config } = e.data;
    try {
        switch (type) {
            case 'init':
                await initModel(config.wasmPath);
                postMessage({ type: 'status', status: 'ready', message: 'Ready' });
                break;
            case 'process':
                if (!isModelLoaded) throw new Error('Model not loaded');
                postMessage({ type: 'status', status: 'processing', message: 'Processing image...' });
                postMessage({ type: 'progress', percent: 20 });
                const result = await processImage(data.imageData, data.width, data.height, data.userMask);
                postMessage({ type: 'status', status: 'processing', message: 'Finalizing...' });
                postMessage({ type: 'progress', percent: 90 });
                postMessage({ type: 'result', result: result, status: 'ready', message: 'Processing complete' }, [result]);
                break;
        }
    } catch (error) {
        console.error('Worker Error:', error);
        postMessage({ type: 'error', message: error.message });
    }
};

async function initModel(wasmPath) {
    if (isModelLoaded) return;
    if (wasmPath) ort.env.wasm.wasmPaths = wasmPath;
    const modelUrl = 'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx';
    // Use only 'wasm' backend - WebGPU requires ES modules which don't work in Web Workers
    modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    modelInputName = modelSession.inputNames[0];
    modelOutputName = modelSession.outputNames[0];
    isModelLoaded = true;
}

function createStarTemplate(size = 48) {
    // PERFORMANCE: Return cached template if available
    if (templateCache.has(size)) {
        return templateCache.get(size);
    }

    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const center = size / 2;
    const outerRadius = size / 2 - 2;
    const innerRadius = size / 6;
    ctx.fillStyle = 'white';
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
        const angle = (i * Math.PI / 4) - Math.PI / 2;
        const r = i % 2 === 0 ? outerRadius : innerRadius;
        const x = center + r * Math.cos(angle);
        const y = center + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    const template = ctx.getImageData(0, 0, size, size);

    // Cache for reuse (limit cache size to prevent memory issues)
    if (templateCache.size < 20) {
        templateCache.set(size, template);
    }

    return template;
}

function resizeTemplate(template, newSize) {
    const canvas = new OffscreenCanvas(newSize, newSize);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const tempCanvas = new OffscreenCanvas(template.width, template.height);
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    tempCtx.putImageData(template, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, newSize, newSize);
    return ctx.getImageData(0, 0, newSize, newSize);
}

// Build integral image (summed-area table) for O(1) rectangular sum queries
function buildIntegralImage(data, w, h) {
    const integral = new Float64Array((w + 1) * (h + 1));
    const integralSq = new Float64Array((w + 1) * (h + 1));
    const stride = w + 1;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const v = data[y * w + x] / 255.0;
            const idx = (y + 1) * stride + (x + 1);
            integral[idx] = v + integral[idx - 1] + integral[idx - stride] - integral[idx - stride - 1];
            integralSq[idx] = v * v + integralSq[idx - 1] + integralSq[idx - stride] - integralSq[idx - stride - 1];
        }
    }
    return { integral, integralSq, stride };
}

function rectSum(integral, stride, x1, y1, x2, y2) {
    return integral[(y2 + 1) * stride + (x2 + 1)]
        - integral[(y1) * stride + (x2 + 1)]
        - integral[(y2 + 1) * stride + (x1)]
        + integral[(y1) * stride + (x1)];
}

function templateMatchNCC(roiGray, roiW, roiH, template, templateSize) {
    const n = templateSize * templateSize;
    const sqrtN = Math.sqrt(n);
    const templateGray = new Float32Array(n);
    let tSum = 0;
    for (let i = 0; i < n; i++) {
        const v = template.data[i * 4] / 255.0;
        templateGray[i] = v;
        tSum += v;
    }
    const tMean = tSum / n;
    let tVar = 0;
    for (let i = 0; i < n; i++) {
        templateGray[i] -= tMean;
        tVar += templateGray[i] * templateGray[i];
    }
    const tNorm = Math.sqrt(tVar);
    if (tNorm < 1e-6) return { maxVal: 0, maxLoc: null };

    // Build integral images for O(1) sum queries
    const { integral, integralSq, stride } = buildIntegralImage(roiGray, roiW, roiH);

    let maxVal = -1;
    let maxLoc = null;

    for (let y = 0; y <= roiH - templateSize; y++) {
        for (let x = 0; x <= roiW - templateSize; x++) {
            // O(1) sum and sumSq using integral images
            const sum = rectSum(integral, stride, x, y, x + templateSize - 1, y + templateSize - 1);
            const sumSq = rectSum(integralSq, stride, x, y, x + templateSize - 1, y + templateSize - 1);

            const mean = sum / n;
            const variance = Math.max(0, sumSq / n - mean * mean);
            const std = Math.sqrt(variance);
            if (std < 1e-6) continue;

            let corr = 0;
            for (let ty = 0; ty < templateSize; ty++) {
                for (let tx = 0; tx < templateSize; tx++) {
                    const idx = (y + ty) * roiW + (x + tx);
                    const v = roiGray[idx] / 255.0 - mean;
                    corr += v * templateGray[ty * templateSize + tx];
                }
            }
            const ncc = corr / (std * sqrtN * tNorm);
            if (ncc > maxVal) {
                maxVal = ncc;
                maxLoc = { x, y };
            }
        }
    }
    return { maxVal, maxLoc };
}

// Detect star mark - Exact port of main.py's detect_star_watermark
function detectStarWatermark(imageDataArr, width, height) {
    // Convert to grayscale (matching cv2.cvtColor)
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const r = imageDataArr[i * 4];
        const g = imageDataArr[i * 4 + 1];
        const b = imageDataArr[i * 4 + 2];
        gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // ROI: Bottom-right 300x300 (matching main.py)
    const searchH = Math.min(300, Math.floor(height / 2));
    const searchW = Math.min(300, Math.floor(width / 2));
    const roiY = height - searchH;
    const roiX = width - searchW;

    // Extract ROI grayscale
    const roiGray = new Uint8Array(searchW * searchH);
    for (let y = 0; y < searchH; y++) {
        for (let x = 0; x < searchW; x++) {
            roiGray[y * searchW + x] = gray[(roiY + y) * width + (roiX + x)];
        }
    }

    const baseTemplate = createStarTemplate(48);

    // Scales: 8 values optimized for common star mark sizes (faster than 10)
    const scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0];

    let bestVal = 0;
    let bestLoc = null;
    let bestSize = null;

    for (const scale of scales) {
        const newSize = Math.round(48 * scale);
        if (newSize < 10 || newSize > Math.min(searchH, searchW)) continue;
        const template = resizeTemplate(baseTemplate, newSize);
        const result = templateMatchNCC(roiGray, searchW, searchH, template, newSize);
        if (result.maxVal > bestVal) {
            bestVal = result.maxVal;
            bestLoc = result.maxLoc;
            bestSize = newSize;
        }
        // Early termination: if we find a very good match, stop searching
        if (bestVal > 0.7) break;
    }

    console.log(`[StarMarkDetect] Best score: ${bestVal.toFixed(3)}`);

    const mask = new Uint8Array(width * height);

    // Threshold 0.35 (matching main.py)
    if (bestVal > 0.35 && bestLoc !== null) {
        const starX = roiX + bestLoc.x;
        const starY = roiY + bestLoc.y;

        // Padding for dilation (matching main.py: pad = max(3, int(best_size * 0.1)))
        const pad = Math.max(3, Math.round(bestSize * 0.1));

        // Create star mask at detected size
        const starTemplate = createStarTemplate(bestSize);

        // Create dilated star mask (simulating cv2.dilate with elliptical kernel)
        const dilatedSize = bestSize + pad * 2;
        const dilatedMask = new Uint8Array(dilatedSize * dilatedSize);

        // For each pixel in the dilated output
        for (let dy = 0; dy < dilatedSize; dy++) {
            for (let dx = 0; dx < dilatedSize; dx++) {
                // Check if any star pixel within elliptical kernel distance
                let found = false;
                for (let ky = -pad; ky <= pad && !found; ky++) {
                    for (let kx = -pad; kx <= pad && !found; kx++) {
                        // Elliptical kernel check
                        if (kx * kx + ky * ky > pad * pad) continue;

                        const sy = dy - pad + ky;
                        const sx = dx - pad + kx;

                        if (sy >= 0 && sy < bestSize && sx >= 0 && sx < bestSize) {
                            if (starTemplate.data[(sy * bestSize + sx) * 4] > 127) {
                                found = true;
                            }
                        }
                    }
                }
                if (found) {
                    dilatedMask[dy * dilatedSize + dx] = 255;
                }
            }
        }

        // Place dilated mask at detected position (matching main.py's mask placement)
        const y1 = starY - pad;
        const x1 = starX - pad;

        for (let dy = 0; dy < dilatedSize; dy++) {
            for (let dx = 0; dx < dilatedSize; dx++) {
                if (dilatedMask[dy * dilatedSize + dx] > 0) {
                    const mx = x1 + dx;
                    const my = y1 + dy;
                    if (mx >= 0 && mx < width && my >= 0 && my < height) {
                        mask[my * width + mx] = 255;
                    }
                }
            }
        }

        console.log(`[StarMarkDetect] DETECTED star at (${starX},${starY}) size ${bestSize}x${bestSize}`);
    } else {
        console.log(`[StarMarkDetect] Score too low (${bestVal.toFixed(3)}), using fallback`);

        // Fallback (matching main.py)
        const starSize = Math.max(20, Math.min(50, Math.round(Math.min(width, height) * 0.02)));
        const centerX = width - 20 - Math.floor(starSize / 2);
        const centerY = height - 100;

        const starTemplate = createStarTemplate(starSize);
        const pad = 5;

        const y1 = Math.max(0, centerY - Math.floor(starSize / 2) - pad);
        const y2 = Math.min(height, centerY + Math.floor(starSize / 2) + pad);
        const x1 = Math.max(0, centerX - Math.floor(starSize / 2) - pad);
        const x2 = Math.min(width, centerX + Math.floor(starSize / 2) + pad);

        for (let y = y1; y < y2; y++) {
            for (let x = x1; x < x2; x++) {
                const sy = y - (centerY - Math.floor(starSize / 2));
                const sx = x - (centerX - Math.floor(starSize / 2));
                if (sy >= 0 && sy < starSize && sx >= 0 && sx < starSize) {
                    if (starTemplate.data[(sy * starSize + sx) * 4] > 127) {
                        mask[y * width + x] = 255;
                    }
                }
            }
        }
    }

    return { mask, detected: bestVal > 0.35 };
}

function reflectIndex(idx, size) {
    if (size <= 1) return 0;
    const period = 2 * (size - 1);
    const mod = idx % period;
    return mod < size ? mod : period - mod;
}

function padImageReflect(srcData, width, height, targetSize) {
    const padded = new Uint8ClampedArray(targetSize * targetSize * 4);
    for (let y = 0; y < targetSize; y++) {
        const sy = reflectIndex(y, height);
        for (let x = 0; x < targetSize; x++) {
            const sx = reflectIndex(x, width);
            const srcIdx = (sy * width + sx) * 4;
            const dstIdx = (y * targetSize + x) * 4;
            padded[dstIdx] = srcData[srcIdx];
            padded[dstIdx + 1] = srcData[srcIdx + 1];
            padded[dstIdx + 2] = srcData[srcIdx + 2];
            padded[dstIdx + 3] = 255;
        }
    }
    return padded;
}

function padMaskReflect(mask, width, height, targetSize) {
    const padded = new Uint8Array(targetSize * targetSize);
    for (let y = 0; y < targetSize; y++) {
        const sy = reflectIndex(y, height);
        for (let x = 0; x < targetSize; x++) {
            const sx = reflectIndex(x, width);
            padded[y * targetSize + x] = mask[sy * width + sx];
        }
    }
    return padded;
}

async function processImage(inputImageData, width, height, userMask = null) {
    const imageData = inputImageData;
    let mask;
    if (userMask) {
        postMessage({ type: 'status', message: 'Processing Custom Mask...' });
        mask = new Uint8Array(width * height);
        for (let i = 0; i < width * height; i++) {
            mask[i] = (userMask[i * 4 + 3] > 0) ? 255 : 0;
        }
    } else {
        postMessage({ type: 'status', message: 'Scanning for star mark...' });
        const detection = detectStarWatermark(imageData, width, height);
        mask = detection.mask;
    }

    let minX = width, minY = height, maxX = 0, maxY = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (mask[y * width + x] > 0) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }
    if (maxX === 0) {
        const c = new OffscreenCanvas(width, height);
        c.getContext('2d').putImageData(new ImageData(imageData, width, height), 0, 0);
        return c.transferToImageBitmap();
    }

    const margin = 64;
    minX = Math.max(0, minX - margin);
    minY = Math.max(0, minY - margin);
    maxX = Math.min(width, maxX + margin);
    maxY = Math.min(height, maxY + margin);
    const cropW = maxX - minX;
    const cropH = maxY - minY;

    const cropMask = new Uint8Array(cropW * cropH);
    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            cropMask[y * cropW + x] = mask[(y + minY) * width + (x + minX)];
        }
    }

    const maxDim = Math.max(cropW, cropH);
    let processW = cropW;
    let processH = cropH;
    if (maxDim > 512) {
        const scale = 512 / maxDim;
        processW = Math.round(cropW * scale);
        processH = Math.round(cropH * scale);
    }

    const processCanvas = new OffscreenCanvas(processW, processH);
    const processCtx = processCanvas.getContext('2d', { willReadFrequently: true });
    const cropCanvas = new OffscreenCanvas(cropW, cropH);
    const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

    const cropData = new Uint8ClampedArray(cropW * cropH * 4);
    for (let y = 0; y < cropH; y++) {
        const rowSrc = (minY + y) * width * 4;
        const rowDst = y * cropW * 4;
        const start = rowSrc + minX * 4;
        const end = start + cropW * 4;
        cropData.set(imageData.subarray(start, end), rowDst);
    }
    cropCtx.putImageData(new ImageData(cropData, cropW, cropH), 0, 0);
    processCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, 0, 0, processW, processH);
    const processData = processCtx.getImageData(0, 0, processW, processH).data;

    const resizedMask = new Uint8Array(processW * processH);
    for (let y = 0; y < processH; y++) {
        for (let x = 0; x < processW; x++) {
            const origX = Math.floor((x / processW) * cropW);
            const origY = Math.floor((y / processH) * cropH);
            resizedMask[y * processW + x] = cropMask[origY * cropW + origX] > 127 ? 1 : 0;
        }
    }

    const targetSize = 512;
    const paddedImage = padImageReflect(processData, processW, processH, targetSize);
    const paddedMask = padMaskReflect(resizedMask, processW, processH, targetSize);

    const inputData = new Float32Array(4 * targetSize * targetSize);
    const area = targetSize * targetSize;
    for (let i = 0; i < area; i++) {
        const dataIdx = i * 4;
        const m = paddedMask[i];
        const r = (paddedImage[dataIdx] / 255.0) * 2 - 1;
        const g = (paddedImage[dataIdx + 1] / 255.0) * 2 - 1;
        const b = (paddedImage[dataIdx + 2] / 255.0) * 2 - 1;
        inputData[i] = 0.5 - m;
        inputData[area + i] = r * (1 - m);
        inputData[2 * area + i] = g * (1 - m);
        inputData[3 * area + i] = b * (1 - m);
    }

    postMessage({ type: 'status', message: 'Removing star mark (AI Inference)...' });
    const inputTensor = new ort.Tensor('float32', inputData, [1, 4, targetSize, targetSize]);
    const feeds = {};
    feeds[modelInputName] = inputTensor;

    const results = await modelSession.run(feeds);
    const output = results[modelOutputName].data;
    const outDims = results[modelOutputName].dims;

    const outputImageData = new ImageData(targetSize, targetSize);
    const outputData = outputImageData.data;
    const isNCHW = outDims?.length === 4 && outDims[1] === 3;

    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const pixelIdx = y * targetSize + x;
            const dataIdx = pixelIdx * 4;
            let rOut, gOut, bOut;
            if (isNCHW) {
                rOut = output[0 * area + pixelIdx];
                gOut = output[1 * area + pixelIdx];
                bOut = output[2 * area + pixelIdx];
            } else {
                rOut = output[pixelIdx * 3];
                gOut = output[pixelIdx * 3 + 1];
                bOut = output[pixelIdx * 3 + 2];
            }
            rOut = Math.min(255, Math.max(0, ((rOut + 1) / 2) * 255));
            gOut = Math.min(255, Math.max(0, ((gOut + 1) / 2) * 255));
            bOut = Math.min(255, Math.max(0, ((bOut + 1) / 2) * 255));
            outputData[dataIdx] = rOut;
            outputData[dataIdx + 1] = gOut;
            outputData[dataIdx + 2] = bOut;
            outputData[dataIdx + 3] = 255;
        }
    }

    const tempResultCanvas = new OffscreenCanvas(targetSize, targetSize);
    const tempResultCtx = tempResultCanvas.getContext('2d');
    tempResultCtx.putImageData(outputImageData, 0, 0);

    const finalCanvas = new OffscreenCanvas(Math.floor(width), Math.floor(height));
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.putImageData(new ImageData(imageData, Math.floor(width), Math.floor(height)), 0, 0);

    const validResultCanvas = new OffscreenCanvas(processW, processH);
    const validCtx = validResultCanvas.getContext('2d');
    validCtx.drawImage(tempResultCanvas, 0, 0, processW, processH, 0, 0, processW, processH);

    const scaledResultCanvas = new OffscreenCanvas(cropW, cropH);
    const scaledResultCtx = scaledResultCanvas.getContext('2d');
    scaledResultCtx.imageSmoothingQuality = 'high';
    scaledResultCtx.drawImage(validResultCanvas, 0, 0, processW, processH, 0, 0, cropW, cropH);
    const inpaintedData = scaledResultCtx.getImageData(0, 0, cropW, cropH).data;

    const originalCropCanvas = new OffscreenCanvas(cropW, cropH);
    const originalCropData = new Uint8ClampedArray(cropW * cropH * 4);
    for (let y = 0; y < cropH; y++) {
        const rowSrc = (minY + y) * width * 4;
        const rowDst = y * cropW * 4;
        const start = rowSrc + minX * 4;
        const end = start + cropW * 4;
        originalCropData.set(imageData.subarray(start, end), rowDst);
    }

    const blurredMask = new Float32Array(cropW * cropH);
    const tempBlur = new Float32Array(cropW * cropH);
    const kSmooth = [0.0625, 0.25, 0.375, 0.25, 0.0625];

    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            let sum = 0;
            const rowOffset = y * cropW;
            for (let k = -2; k <= 2; k++) {
                const px = Math.min(Math.max(x + k, 0), cropW - 1);
                const val = (cropMask[rowOffset + px] > 127) ? 1.0 : 0.0;
                sum += val * kSmooth[k + 2];
            }
            tempBlur[rowOffset + x] = sum;
        }
    }
    for (let x = 0; x < cropW; x++) {
        for (let y = 0; y < cropH; y++) {
            let sum = 0;
            for (let k = -2; k <= 2; k++) {
                const py = Math.min(Math.max(y + k, 0), cropH - 1);
                sum += tempBlur[py * cropW + x] * kSmooth[k + 2];
            }
            blurredMask[y * cropW + x] = sum;
        }
    }

    const blendedData = new Uint8ClampedArray(cropW * cropH * 4);
    for (let i = 0; i < cropW * cropH; i++) {
        const m = blurredMask[i];
        const invM = 1.0 - m;
        const idx = i * 4;
        blendedData[idx] = (inpaintedData[idx] * m + originalCropData[idx] * invM);
        blendedData[idx + 1] = (inpaintedData[idx + 1] * m + originalCropData[idx + 1] * invM);
        blendedData[idx + 2] = (inpaintedData[idx + 2] * m + originalCropData[idx + 2] * invM);
        blendedData[idx + 3] = 255;
    }

    const blendedImageData = new ImageData(blendedData, cropW, cropH);
    finalCtx.putImageData(blendedImageData, minX, minY);

    return finalCanvas.transferToImageBitmap();
}

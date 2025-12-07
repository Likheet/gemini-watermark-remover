// Offscreen document for Star Mark Remover
// Handles heavy image processing in a separate context to avoid UI freezing

let modelSession = null;
let isModelLoaded = false;
let modelInputName = null;
let modelOutputName = null;

// Suppress "Unknown CPU vendor" logs
const originalWarn = console.warn;
const originalLog = console.log;
const filterFn = (args) => {
    const msg = args[0];
    return typeof msg === 'string' && (msg.includes('Unknown CPU vendor') || msg.includes('cpuinfo_vendor value'));
};
console.warn = function (...args) { if (!filterFn(args)) originalWarn.apply(console, args); };
console.log = function (...args) { if (!filterFn(args)) originalLog.apply(console, args); };

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'processImage') {
        processImage(message.imageData, message.width, message.height)
            .then(result => {
                sendResponse(result);
            })
            .catch(err => {
                sendResponse({ error: err.message });
            });
        return true; // Keep channel open for async response
    }
});

async function initModel() {
    if (isModelLoaded) return;

    ort.env.logLevel = 'error';
    ort.env.wasm.numThreads = 1;

    const modelUrl = 'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx';
    modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    modelInputName = modelSession.inputNames[0];
    modelOutputName = modelSession.outputNames[0];
    isModelLoaded = true;
    console.log('[Offscreen] Model loaded');
}

// Star template creation
function createStarTemplate(size = 48) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const center = size / 2;
    const outerRadius = size / 2 - 2;
    const innerRadius = outerRadius * 0.4;

    ctx.fillStyle = 'white';
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
        const angle = (Math.PI / 4) * i - Math.PI / 2;
        const radius = i % 2 === 0 ? outerRadius : innerRadius;
        const x = center + Math.cos(angle) * radius;
        const y = center + Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    return ctx.getImageData(0, 0, size, size);
}

function resizeTemplate(template, newSize) {
    const canvas = document.createElement('canvas');
    canvas.width = newSize;
    canvas.height = newSize;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const imgData = ctx.createImageData(newSize, newSize);
    for (let y = 0; y < newSize; y++) {
        for (let x = 0; x < newSize; x++) {
            const srcX = Math.floor((x / newSize) * template.width);
            const srcY = Math.floor((y / newSize) * template.height);
            const srcIdx = (srcY * template.width + srcX) * 4;
            const dstIdx = (y * newSize + x) * 4;
            imgData.data[dstIdx] = template.data[srcIdx];
            imgData.data[dstIdx + 1] = template.data[srcIdx + 1];
            imgData.data[dstIdx + 2] = template.data[srcIdx + 2];
            imgData.data[dstIdx + 3] = template.data[srcIdx + 3];
        }
    }
    return imgData;
}

// Integral image for O(1) sum queries
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
    const a = integral[(y1) * stride + (x1)];
    const b = integral[(y1) * stride + (x2 + 1)];
    const c = integral[(y2 + 1) * stride + (x1)];
    const d = integral[(y2 + 1) * stride + (x2 + 1)];
    return d - b - c + a;
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

    const { integral, integralSq, stride } = buildIntegralImage(roiGray, roiW, roiH);
    let maxVal = -1;
    let maxLoc = null;

    for (let y = 0; y <= roiH - templateSize; y++) {
        for (let x = 0; x <= roiW - templateSize; x++) {
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

function detectStarWatermark(imageData, width, height) {
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];
        gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const searchH = Math.min(300, Math.floor(height / 2));
    const searchW = Math.min(300, Math.floor(width / 2));
    const roiY = height - searchH;
    const roiX = width - searchW;

    const roiGray = new Uint8Array(searchW * searchH);
    for (let y = 0; y < searchH; y++) {
        for (let x = 0; x < searchW; x++) {
            roiGray[y * searchW + x] = gray[(roiY + y) * width + (roiX + x)];
        }
    }

    const baseTemplate = createStarTemplate(48);
    const scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0];
    let bestVal = 0, bestLoc = null, bestSize = null;

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
        if (bestVal > 0.7) break;
    }

    console.log(`[Offscreen] Best score: ${bestVal.toFixed(3)}`);
    const mask = new Uint8Array(width * height);

    if (bestVal > 0.35 && bestLoc !== null) {
        const starX = roiX + bestLoc.x;
        const starY = roiY + bestLoc.y;
        const pad = Math.max(3, Math.round(bestSize * 0.1));
        const starTemplate = createStarTemplate(bestSize);
        const dilatedSize = bestSize + pad * 2;
        const dilatedMask = new Uint8Array(dilatedSize * dilatedSize);

        for (let dy = 0; dy < dilatedSize; dy++) {
            for (let dx = 0; dx < dilatedSize; dx++) {
                let found = false;
                for (let ky = -pad; ky <= pad && !found; ky++) {
                    for (let kx = -pad; kx <= pad && !found; kx++) {
                        if (kx * kx + ky * ky > pad * pad) continue;
                        const sy = dy - pad + ky;
                        const sx = dx - pad + kx;
                        if (sy >= 0 && sy < bestSize && sx >= 0 && sx < bestSize) {
                            if (starTemplate.data[(sy * bestSize + sx) * 4] > 127) found = true;
                        }
                    }
                }
                if (found) dilatedMask[dy * dilatedSize + dx] = 255;
            }
        }

        const y1 = starY - pad;
        const x1 = starX - pad;
        for (let dy = 0; dy < dilatedSize; dy++) {
            for (let dx = 0; dx < dilatedSize; dx++) {
                if (dilatedMask[dy * dilatedSize + dx] > 0) {
                    const mx = x1 + dx, my = y1 + dy;
                    if (mx >= 0 && mx < width && my >= 0 && my < height) {
                        mask[my * width + mx] = 255;
                    }
                }
            }
        }
        console.log(`[Offscreen] DETECTED star at (${starX},${starY}) size ${bestSize}x${bestSize}`);
    } else {
        console.log(`[Offscreen] Score too low, using fallback`);
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
    return { mask };
}

function reflectIndex(idx, size) {
    if (idx < 0) return -idx - 1;
    if (idx >= size) return 2 * size - idx - 1;
    return idx;
}

function padImageReflect(srcData, width, height, targetSize) {
    const padLeft = Math.floor((targetSize - width) / 2);
    const padTop = Math.floor((targetSize - height) / 2);
    const result = new Uint8ClampedArray(targetSize * targetSize * 4);
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcY = reflectIndex(y - padTop, height);
            const srcX = reflectIndex(x - padLeft, width);
            const srcIdx = (srcY * width + srcX) * 4;
            const dstIdx = (y * targetSize + x) * 4;
            result[dstIdx] = srcData[srcIdx];
            result[dstIdx + 1] = srcData[srcIdx + 1];
            result[dstIdx + 2] = srcData[srcIdx + 2];
            result[dstIdx + 3] = srcData[srcIdx + 3];
        }
    }
    return result;
}

function padMaskReflect(mask, width, height, targetSize) {
    const padLeft = Math.floor((targetSize - width) / 2);
    const padTop = Math.floor((targetSize - height) / 2);
    const result = new Uint8Array(targetSize * targetSize);
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcY = reflectIndex(y - padTop, height);
            const srcX = reflectIndex(x - padLeft, width);
            result[y * targetSize + x] = mask[srcY * width + srcX];
        }
    }
    return result;
}

async function processImage(inputImageData, width, height) {
    await initModel();

    const imageData = new Uint8ClampedArray(inputImageData);
    const detection = detectStarWatermark(imageData, width, height);
    const mask = detection.mask;

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
        // No mask detected, return original
        return { imageData: Array.from(imageData), width, height, noMask: true };
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
    let processW = cropW, processH = cropH;
    if (maxDim > 512) {
        const scale = 512 / maxDim;
        processW = Math.round(cropW * scale);
        processH = Math.round(cropH * scale);
    }

    // Extract and resize crop
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
    const cropData = new Uint8ClampedArray(cropW * cropH * 4);
    for (let y = 0; y < cropH; y++) {
        const rowSrc = (minY + y) * width * 4;
        const rowDst = y * cropW * 4;
        cropData.set(imageData.subarray(rowSrc + minX * 4, rowSrc + minX * 4 + cropW * 4), rowDst);
    }
    cropCtx.putImageData(new ImageData(cropData, cropW, cropH), 0, 0);

    const processCanvas = document.createElement('canvas');
    processCanvas.width = processW;
    processCanvas.height = processH;
    const processCtx = processCanvas.getContext('2d', { willReadFrequently: true });
    processCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, 0, 0, processW, processH);
    const processData = processCtx.getImageData(0, 0, processW, processH).data;
    const originalCropData = new Uint8ClampedArray(cropData);

    // Resize mask
    const resizedMask = new Uint8Array(processW * processH);
    for (let y = 0; y < processH; y++) {
        for (let x = 0; x < processW; x++) {
            const origX = Math.floor((x / processW) * cropW);
            const origY = Math.floor((y / processH) * cropH);
            resizedMask[y * processW + x] = cropMask[origY * cropW + origX];
        }
    }

    // Pad to 512x512
    const targetSize = 512;
    const paddedImage = padImageReflect(processData, processW, processH, targetSize);
    const paddedMask = padMaskReflect(resizedMask, processW, processH, targetSize);

    // Prepare input tensor
    const area = targetSize * targetSize;
    const inputData = new Float32Array(4 * area);
    for (let i = 0; i < area; i++) {
        const r = paddedImage[i * 4] / 255.0 * 2 - 1;
        const g = paddedImage[i * 4 + 1] / 255.0 * 2 - 1;
        const b = paddedImage[i * 4 + 2] / 255.0 * 2 - 1;
        const m = paddedMask[i] / 255.0;
        inputData[i] = r * (1 - m);
        inputData[area + i] = g * (1 - m);
        inputData[2 * area + i] = b * (1 - m);
        inputData[3 * area + i] = m;
    }

    // Run inference
    console.log('[Offscreen] Running MI-GAN inference...');
    const inputTensor = new ort.Tensor('float32', inputData, [1, 4, targetSize, targetSize]);
    const feeds = {};
    feeds[modelInputName] = inputTensor;
    const results = await modelSession.run(feeds);
    const output = results[modelOutputName].data;
    console.log('[Offscreen] Inference complete');

    // Process output
    const outputData = new Uint8ClampedArray(targetSize * targetSize * 4);
    for (let i = 0; i < area; i++) {
        outputData[i * 4] = Math.round(Math.max(0, Math.min(255, ((output[i] + 1) / 2) * 255)));
        outputData[i * 4 + 1] = Math.round(Math.max(0, Math.min(255, ((output[area + i] + 1) / 2) * 255)));
        outputData[i * 4 + 2] = Math.round(Math.max(0, Math.min(255, ((output[2 * area + i] + 1) / 2) * 255)));
        outputData[i * 4 + 3] = 255;
    }

    // Unpad and resize back
    const inpaintedCanvas = document.createElement('canvas');
    inpaintedCanvas.width = targetSize;
    inpaintedCanvas.height = targetSize;
    const inpaintedCtx = inpaintedCanvas.getContext('2d', { willReadFrequently: true });
    inpaintedCtx.putImageData(new ImageData(outputData, targetSize, targetSize), 0, 0);

    const backCanvas = document.createElement('canvas');
    backCanvas.width = cropW;
    backCanvas.height = cropH;
    const backCtx = backCanvas.getContext('2d', { willReadFrequently: true });
    backCtx.drawImage(inpaintedCanvas, 0, 0, processW, processH, 0, 0, cropW, cropH);
    const inpaintedData = backCtx.getImageData(0, 0, cropW, cropH).data;

    // Blur mask for blending
    const blurredMask = new Float32Array(cropW * cropH);
    const kSmooth = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136];
    const tempBlur = new Float32Array(cropW * cropH);
    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            let sum = 0;
            for (let k = -2; k <= 2; k++) {
                const px = Math.min(Math.max(x + k, 0), cropW - 1);
                sum += (cropMask[y * cropW + px] / 255.0) * kSmooth[k + 2];
            }
            tempBlur[y * cropW + x] = sum;
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

    // Blend
    const blendedData = new Uint8ClampedArray(cropW * cropH * 4);
    for (let i = 0; i < cropW * cropH; i++) {
        const m = blurredMask[i];
        const invM = 1.0 - m;
        const idx = i * 4;
        blendedData[idx] = Math.round(inpaintedData[idx] * m + originalCropData[idx] * invM);
        blendedData[idx + 1] = Math.round(inpaintedData[idx + 1] * m + originalCropData[idx + 1] * invM);
        blendedData[idx + 2] = Math.round(inpaintedData[idx + 2] * m + originalCropData[idx + 2] * invM);
        blendedData[idx + 3] = 255;
    }

    // Build final image
    const finalData = new Uint8ClampedArray(imageData);
    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            const srcIdx = (y * cropW + x) * 4;
            const dstIdx = ((minY + y) * width + (minX + x)) * 4;
            finalData[dstIdx] = blendedData[srcIdx];
            finalData[dstIdx + 1] = blendedData[srcIdx + 1];
            finalData[dstIdx + 2] = blendedData[srcIdx + 2];
            finalData[dstIdx + 3] = 255;
        }
    }

    console.log('[Offscreen] Processing complete');
    return { imageData: Array.from(finalData), width, height, noMask: false };
}

console.log('[Offscreen] Document ready');

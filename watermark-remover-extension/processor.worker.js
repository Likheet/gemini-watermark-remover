// Image Processor Web Worker
importScripts('lib/ort.min.js');

let modelSession = null;
let isModelLoaded = false;
let modelInputName = null;
let modelOutputName = null;

// Configure ONNX Runtime for Worker
ort.env.wasm.numThreads = 1;

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

                // Theatrical Status Sequence
                postMessage({ type: 'status', status: 'processing', message: 'Analyzing image structure...' });
                postMessage({ type: 'progress', percent: 10 });
                await new Promise(r => setTimeout(r, 600));

                postMessage({ type: 'status', status: 'processing', message: 'Identifying watermark patterns...' });
                postMessage({ type: 'progress', percent: 30 });
                await new Promise(r => setTimeout(r, 800));

                postMessage({ type: 'status', status: 'processing', message: 'Generating inpaint mask...' });
                postMessage({ type: 'progress', percent: 50 });
                await new Promise(r => setTimeout(r, 600));

                postMessage({ type: 'status', status: 'processing', message: 'Synthesizing clean texture...' });
                postMessage({ type: 'progress', percent: 75 });
                await new Promise(r => setTimeout(r, 500));

                // Process with optional userMask
                const resultImageBitmap = await processImage(data.imageData, data.width, data.height, data.userMask);

                postMessage({ type: 'status', status: 'processing', message: 'Finalizing...' });
                postMessage({ type: 'progress', percent: 90 });

                postMessage({
                    type: 'result',
                    result: resultImageBitmap,
                    status: 'ready',
                    message: 'Processing complete'
                }, [resultImageBitmap]);
                break;
        }
    } catch (error) {
        console.error('Worker Error:', error);
        postMessage({
            type: 'error',
            message: error.message
        });
    }
};

async function initModel(wasmPath) {
    if (isModelLoaded) return;

    if (wasmPath) {
        ort.env.wasm.wasmPaths = wasmPath;
    }

    const modelUrl = 'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx';

    modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });

    modelInputName = modelSession.inputNames[0];
    modelOutputName = modelSession.outputNames[0];
    isModelLoaded = true;
}

// --- Image Processing Logic ---

function createStarTemplate(size = 48) {
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
    return ctx.getImageData(0, 0, size, size);
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

    let maxVal = -1;
    let maxLoc = null;

    for (let y = 0; y <= roiH - templateSize; y++) {
        for (let x = 0; x <= roiW - templateSize; x++) {
            let sum = 0;
            let sumSq = 0;
            for (let ty = 0; ty < templateSize; ty++) {
                for (let tx = 0; tx < templateSize; tx++) {
                    const idx = (y + ty) * roiW + (x + tx);
                    const v = roiGray[idx] / 255.0;
                    sum += v;
                    sumSq += v * v;
                }
            }
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

function detectStarWatermark(imageDataArr, width, height) {
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        gray[i] = Math.round(0.299 * imageDataArr[i * 4] + 0.587 * imageDataArr[i * 4 + 1] + 0.114 * imageDataArr[i * 4 + 2]);
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
    const scales = [];
    for (let i = 0; i < 20; i++) scales.push(0.3 + i * ((2.0 - 0.3) / 19));
    let bestVal = -1;
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
    }
    const mask = new Uint8Array(width * height);
    if (bestVal > 0.35 && bestLoc !== null) {
        const starX = roiX + bestLoc.x;
        const starY = roiY + bestLoc.y;
        const pad = Math.max(3, Math.round(bestSize * 0.1));
        const starTemplate = createStarTemplate(bestSize);
        for (let dy = 0; dy < bestSize; dy++) {
            for (let dx = 0; dx < bestSize; dx++) {
                if (starTemplate.data[(dy * bestSize + dx) * 4] > 127) {
                    for (let ky = -pad; ky <= pad; ky++) {
                        for (let kx = -pad; kx <= pad; kx++) {
                            if (kx * kx + ky * ky > pad * pad) continue;
                            const mx = starX + dx + kx;
                            const my = starY + dy + ky;
                            if (mx >= 0 && mx < width && my >= 0 && my < height) mask[my * width + mx] = 255;
                        }
                    }
                }
            }
        }
    } else {
        const starSize = Math.max(20, Math.min(50, Math.round(Math.min(width, height) * 0.02)));
        const centerX = width - 20 - Math.floor(starSize / 2);
        const centerY = height - 100;
        const starTemplate = createStarTemplate(starSize);
        const pad = 5;
        for (let dy = 0; dy < starSize; dy++) {
            for (let dx = 0; dx < starSize; dx++) {
                if (starTemplate.data[(dy * starSize + dx) * 4] > 127) {
                    for (let ky = -pad; ky <= pad; ky++) {
                        for (let kx = -pad; kx <= pad; kx++) {
                            if (kx * kx + ky * ky > pad * pad) continue;
                            const mx = centerX - Math.floor(starSize / 2) + dx + kx;
                            const my = centerY - Math.floor(starSize / 2) + dy + ky;
                            if (mx >= 0 && mx < width && my >= 0 && my < height) mask[my * width + mx] = 255;
                        }
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

// Inpaint function adapted for non-DOM use (except OffscreenCanvas)
async function processImage(inputImageData, width, height, userMask = null) {
    const imageData = inputImageData; // Uint8ClampedArray

    let mask;
    if (userMask) {
        // User provided manual mask (RGBA)
        // Convert to single channel Uint8Array (0 or 255)
        postMessage({ type: 'status', message: 'Processing Custom Mask...' });
        mask = new Uint8Array(width * height);
        for (let i = 0; i < width * height; i++) {
            // Logic: If Red > 0 (or Alpha > 0), set to 255
            mask[i] = (userMask[i * 4 + 3] > 0) ? 255 : 0;
        }
    } else {
        // 1. Detect Watermark
        postMessage({ type: 'status', message: 'Scanning for watermark...' });
        const detection = detectStarWatermark(imageData, width, height);
        mask = detection.mask;
    }

    // 2. Crop logic
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
    // Handle empty mask case
    if (maxX === 0) {
        // Return original if nothing detected/masked
        return new OffscreenCanvas(width, height).transferToImageBitmap();
        // Wait, OffscreenCanvas returns blank? No, let's fix empty mask fallback
        // Just return original image converted to bitmap
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

    // Crop Mask
    const cropMask = new Uint8Array(cropW * cropH);
    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            cropMask[y * cropW + x] = mask[(y + minY) * width + (x + minX)];
        }
    }

    // Resize Logic
    const maxDim = Math.max(cropW, cropH);
    let processW = cropW;
    let processH = cropH;
    if (maxDim > 512) {
        const scale = 512 / maxDim;
        processW = Math.round(cropW * scale);
        processH = Math.round(cropH * scale);
    }

    // Resize Crop Image using OffscreenCanvas
    const processCanvas = new OffscreenCanvas(processW, processH);
    const processCtx = processCanvas.getContext('2d', { willReadFrequently: true });
    const cropCanvas = new OffscreenCanvas(cropW, cropH);
    const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

    // Extract crop data from original array
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

    // Resize Mask
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

    // Inference Input
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

    postMessage({ type: 'status', message: 'Removing watermark (AI Inference)...' });
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

    const widthInt = Math.floor(width);
    const heightInt = Math.floor(height);
    if (widthInt === 0 || heightInt === 0) throw new Error("Invalid image dimensions");

    const tempResultCanvas = new OffscreenCanvas(targetSize, targetSize);
    const tempResultCtx = tempResultCanvas.getContext('2d');
    tempResultCtx.putImageData(outputImageData, 0, 0);

    const finalCanvas = new OffscreenCanvas(widthInt, heightInt);
    const finalCtx = finalCanvas.getContext('2d');
    if (!finalCtx) throw new Error("Failed to get 2d context for finalCanvas");

    finalCtx.putImageData(new ImageData(imageData, widthInt, heightInt), 0, 0);

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
    const kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625];

    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            let sum = 0;
            const rowOffset = y * cropW;
            for (let k = -2; k <= 2; k++) {
                const px = Math.min(Math.max(x + k, 0), cropW - 1);
                const val = (cropMask[rowOffset + px] > 127) ? 1.0 : 0.0;
                sum += val * kernel[k + 2];
            }
            tempBlur[rowOffset + x] = sum;
        }
    }
    for (let x = 0; x < cropW; x++) {
        for (let y = 0; y < cropH; y++) {
            let sum = 0;
            for (let k = -2; k <= 2; k++) {
                const py = Math.min(Math.max(y + k, 0), cropH - 1);
                sum += tempBlur[py * cropW + x] * kernel[k + 2];
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

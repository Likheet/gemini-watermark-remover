// Content script for Gemini Watermark Remover
// Handles image processing on web pages - Ported from Python MI-GAN implementation

let modelSession = null;
let isModelLoaded = false;
let isModelLoading = false;
let modelInputName = null;
let modelOutputName = null;

// Enable to download a visualized mask overlay for debugging
const DEBUG_MASK_DOWNLOAD = false;

// Check if ONNX Runtime is available (loaded via manifest)
function checkONNXRuntime() {
  if (typeof ort !== 'undefined') {
    console.log('[WatermarkRemover] ONNX Runtime is available');
    return true;
  }

  console.error('[WatermarkRemover] ONNX Runtime not loaded');

  return false;

}



// Load the MI-GAN model

async function loadModel() {

  if (isModelLoaded && modelSession) return;

  if (isModelLoading) {

    // Wait for existing load to complete

    while (isModelLoading) {

      await new Promise(r => setTimeout(r, 100));

    }

    if (isModelLoaded) return;

  }



  isModelLoading = true;



  try {

    showNotification('Loading AI model... (~30MB)');



    if (!checkONNXRuntime()) {

      throw new Error('ONNX Runtime not available');

    }



    // Configure WASM paths to use extension's bundled files

    const wasmPath = chrome.runtime.getURL('lib/');

    ort.env.wasm.wasmPaths = wasmPath;

    // Use single-threaded mode to avoid cross-origin isolation requirement

    ort.env.wasm.numThreads = 1;

    console.log('[WatermarkRemover] WASM path:', wasmPath);



    const modelUrl = 'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx';



    console.log('[WatermarkRemover] Loading MI-GAN model from:', modelUrl);

    modelSession = await ort.InferenceSession.create(modelUrl, {

      executionProviders: ['wasm'],

      graphOptimizationLevel: 'all'

    });



    modelInputName = modelSession.inputNames[0];

    modelOutputName = modelSession.outputNames[0];



    console.log('[WatermarkRemover] Model input name:', modelInputName);

    console.log('[WatermarkRemover] Model output name:', modelOutputName);



    isModelLoaded = true;

    isModelLoading = false;

    console.log('[WatermarkRemover] MI-GAN model loaded successfully!');

    showNotification('AI model loaded!');

  } catch (error) {

    console.error('[WatermarkRemover] Failed to load model:', error);

    isModelLoaded = false;

    isModelLoading = false;

    modelSession = null;

    throw new Error('Failed to load AI model: ' + (error.message || String(error)));

  }

}



// Create a 4-pointed star template for template matching (from Python code)

function createStarTemplate(size = 48) {

  const canvas = document.createElement('canvas');

  canvas.width = size;

  canvas.height = size;

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



    if (i === 0) {

      ctx.moveTo(x, y);

    } else {

      ctx.lineTo(x, y);

    }

  }



  ctx.closePath();

  ctx.fill();



  return ctx.getImageData(0, 0, size, size);

}



// Resize template using nearest neighbor

function resizeTemplate(template, newSize) {

  const canvas = document.createElement('canvas');

  canvas.width = newSize;

  canvas.height = newSize;

  const ctx = canvas.getContext('2d', { willReadFrequently: true });



  const tempCanvas = document.createElement('canvas');

  tempCanvas.width = template.width;

  tempCanvas.height = template.height;

  const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });

  tempCtx.putImageData(template, 0, 0);



  ctx.imageSmoothingEnabled = false;

  ctx.drawImage(tempCanvas, 0, 0, newSize, newSize);



  return ctx.getImageData(0, 0, newSize, newSize);

}



// Template matching using normalized cross-correlation (cv2.TM_CCOEFF_NORMED equivalent)
function templateMatchNCC(roiGray, roiW, roiH, template, templateSize) {
  const n = templateSize * templateSize;
  const sqrtN = Math.sqrt(n);

  // Template stats (grayscale 0..1)
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

  // tStd in orig code was actually ||T'|| (L2 norm)
  const tNorm = Math.sqrt(tVar);
  if (tNorm < 1e-6) return { maxVal: 0, maxLoc: null };

  let maxVal = -1;
  let maxLoc = null;

  for (let y = 0; y <= roiH - templateSize; y++) {
    for (let x = 0; x <= roiW - templateSize; x++) {
      let sum = 0;
      let sumSq = 0;

      // Mean/std for this window
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
      const std = Math.sqrt(variance); // sigma_I
      if (std < 1e-6) continue;

      // Correlation with zero-mean template/window
      let corr = 0;
      for (let ty = 0; ty < templateSize; ty++) {
        for (let tx = 0; tx < templateSize; tx++) {
          const idx = (y + ty) * roiW + (x + tx);
          const v = roiGray[idx] / 255.0 - mean;
          corr += v * templateGray[ty * templateSize + tx];
        }
      }

      // cv2.TM_CCOEFF_NORMED: dot(T', I') / (|T'| * |I'|)
      // |I'| = std * sqrt(n)
      const ncc = corr / (std * sqrtN * tNorm);

      if (ncc > maxVal) {
        maxVal = ncc;
        maxLoc = { x, y };
      }
    }
  }

  return { maxVal, maxLoc };
}

// Detect star watermark using template matching (ported from Python)

function detectStarWatermark(imageData, width, height) {
  console.log(`[WatermarkRemover] Image size: ${width}x${height}`);

  // Convert to grayscale
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = imageData[i * 4];
    const g = imageData[i * 4 + 1];
    const b = imageData[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  // Focus on bottom-right area
  const searchH = Math.min(300, Math.floor(height / 2));
  const searchW = Math.min(300, Math.floor(width / 2));
  const roiY = height - searchH;
  const roiX = width - searchW;

  console.log(`[WatermarkRemover] Searching in bottom-right region: (${roiX},${roiY}) to (${width},${height})`);

  // Extract ROI grayscale for faster access
  const roiGray = new Uint8Array(searchW * searchH);
  for (let y = 0; y < searchH; y++) {
    for (let x = 0; x < searchW; x++) {
      roiGray[y * searchW + x] = gray[(roiY + y) * width + (roiX + x)];
    }
  }

  // Multi-scale template matching (np.linspace(0.3, 2.0, 20))
  const baseTemplate = createStarTemplate(48);
  const scales = [];
  for (let i = 0; i < 20; i++) {
    scales.push(0.3 + i * ((2.0 - 0.3) / 19));
  }

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

  console.log(`[WatermarkRemover] Template matching best score: ${bestVal.toFixed(3)}`);

  const mask = new Uint8Array(width * height);

  if (bestVal > 0.35 && bestLoc !== null) {
    const starX = roiX + bestLoc.x;
    const starY = roiY + bestLoc.y;
    const pad = Math.max(3, Math.round(bestSize * 0.1));

    console.log(`[WatermarkRemover] DETECTED star at (${starX},${starY}) size ${bestSize}x${bestSize}`);

    const starTemplate = createStarTemplate(bestSize);

    for (let dy = 0; dy < bestSize; dy++) {
      for (let dx = 0; dx < bestSize; dx++) {
        const tIdx = (dy * bestSize + dx) * 4;
        if (starTemplate.data[tIdx] > 127) {
          for (let ky = -pad; ky <= pad; ky++) {
            for (let kx = -pad; kx <= pad; kx++) {
              if (kx * kx + ky * ky > pad * pad) continue; // circular-ish dilation
              const mx = starX + dx + kx;
              const my = starY + dy + ky;
              if (mx >= 0 && mx < width && my >= 0 && my < height) {
                mask[my * width + mx] = 255;
              }
            }
          }
        }
      }
    }
  } else {
    console.log(`[WatermarkRemover] Template matching confidence too low (${bestVal.toFixed(3)}), using fallback...`);

    const starSize = Math.max(20, Math.min(50, Math.round(Math.min(width, height) * 0.02)));
    const centerX = width - 20 - Math.floor(starSize / 2);
    const centerY = height - 100;

    const starTemplate = createStarTemplate(starSize);
    const pad = 5;

    for (let dy = 0; dy < starSize; dy++) {
      for (let dx = 0; dx < starSize; dx++) {
        const tIdx = (dy * starSize + dx) * 4;
        if (starTemplate.data[tIdx] > 127) {
          for (let ky = -pad; ky <= pad; ky++) {
            for (let kx = -pad; kx <= pad; kx++) {
              if (kx * kx + ky * ky > pad * pad) continue;
              const mx = centerX - Math.floor(starSize / 2) + dx + kx;
              const my = centerY - Math.floor(starSize / 2) + dy + ky;
              if (mx >= 0 && mx < width && my >= 0 && my < height) {
                mask[my * width + mx] = 255;
              }
            }
          }
        }
      }
    }

    console.log(`[WatermarkRemover] Using fallback position: center=(${centerX},${centerY})`);
  }

  const detectedPixels = mask.reduce((sum, v) => sum + (v > 0 ? 1 : 0), 0);
  console.log(`[WatermarkRemover] Mask covers ${detectedPixels} pixels`);

  return { mask, detected: bestVal > 0.35 };
}

// Separable Gaussian Blur (approximates cv2.GaussianBlur with 5x5 kernel)
// Kernel: [1, 4, 6, 4, 1] / 16
function blurMask(mask, width, height, radius) {
  const blurred = new Float32Array(width * height);
  const temp = new Float32Array(width * height);

  const kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  const halfK = 2; // Kernel size 5, center at 2

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -halfK; k <= halfK; k++) {
        const px = Math.min(Math.max(x + k, 0), width - 1);
        sum += mask[y * width + px] * kernel[k + halfK];
      }
      temp[y * width + x] = sum;
    }
  }

  // Vertical pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -halfK; k <= halfK; k++) {
        const py = Math.min(Math.max(y + k, 0), height - 1);
        sum += temp[py * width + x] * kernel[k + halfK];
      }
      blurred[y * width + x] = sum;
    }
  }

  return blurred;
}

// Reflective padding helpers (mimic numpy.pad(..., mode='reflect'))

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

      padded[dstIdx + 3] = 255; // Force opaque alpha

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



// Run MI-GAN inpainting (ported from Python MiganInpainter)

async function inpaint(imageCanvas) {
  if (!isModelLoaded || !modelSession) {
    await loadModel();
  }

  const ctx = imageCanvas.getContext('2d', { willReadFrequently: true });

  const width = imageCanvas.width;

  const height = imageCanvas.height;

  const imageDataObj = ctx.getImageData(0, 0, width, height);

  const imageData = imageDataObj.data;


  const detection = detectStarWatermark(imageData, width, height);
  const mask = detection.mask;

  // stash mask + image for optional debugging download
  window.__lastMaskData = {
    width,
    height,
    mask,
    imageData: imageData.slice()
  };

  let maskSum = 0;
  for (let i = 0; i < mask.length; i++) maskSum += mask[i];
  if (maskSum === 0) {
    console.warn('[WatermarkRemover] No mask detected, returning original');
    showNotification('No watermark detected');
    return imageCanvas;

  }



  // Get crop region with margin (like Python _get_crop_region)

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



  const margin = 64;

  minX = Math.max(0, minX - margin);

  minY = Math.max(0, minY - margin);

  maxX = Math.min(width, maxX + margin);

  maxY = Math.min(height, maxY + margin);



  const cropW = maxX - minX;

  const cropH = maxY - minY;



  console.log(`[WatermarkRemover] Crop region: (${minX},${minY}) to (${maxX},${maxY}), size: ${cropW}x${cropH}`);



  // Extract crop mask EARLY for later blending

  const cropMask = new Uint8Array(cropW * cropH);

  for (let y = 0; y < cropH; y++) {

    for (let x = 0; x < cropW; x++) {

      cropMask[y * cropW + x] = mask[(y + minY) * width + (x + minX)];

    }

  }



  // Capture original crop for blending later

  const originalCrop = ctx.getImageData(minX, minY, cropW, cropH);



  // Resize crop to max 512 while keeping aspect ratio (matches Python scaling)

  const maxDim = Math.max(cropW, cropH);

  let processW = cropW;

  let processH = cropH;

  if (maxDim > 512) {

    const scale = 512 / maxDim;

    processW = Math.round(cropW * scale);

    processH = Math.round(cropH * scale);

  }



  // Draw resized crop

  const processCanvas = document.createElement('canvas');

  processCanvas.width = processW;

  processCanvas.height = processH;

  const processCtx = processCanvas.getContext('2d', { willReadFrequently: true });

  processCtx.drawImage(imageCanvas, minX, minY, cropW, cropH, 0, 0, processW, processH);

  const processData = processCtx.getImageData(0, 0, processW, processH).data;



  // Resize mask to processing size

  const resizedMask = new Uint8Array(processW * processH);

  for (let y = 0; y < processH; y++) {

    for (let x = 0; x < processW; x++) {

      const origX = Math.floor((x / processW) * cropW);

      const origY = Math.floor((y / processH) * cropH);

      resizedMask[y * processW + x] = cropMask[origY * cropW + origX] > 127 ? 1 : 0;

    }

  }



  // Pad to 512x512 with reflect (aligns with numpy.pad in Python)

  const targetSize = 512;

  const paddedImage = padImageReflect(processData, processW, processH, targetSize);

  const paddedMask = padMaskReflect(resizedMask, processW, processH, targetSize);



  // Create MI-GAN input: [0.5 - mask, erased_R, erased_G, erased_B]

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



  console.log('[WatermarkRemover] Running MI-GAN inference...');

  const inputTensor = new ort.Tensor('float32', inputData, [1, 4, targetSize, targetSize]);



  const feeds = {};

  feeds[modelInputName] = inputTensor;



  const results = await modelSession.run(feeds);

  const output = results[modelOutputName].data;



  const outDims = results[modelOutputName].dims;

  console.log('[WatermarkRemover] Inference complete, output shape:', outDims);



  // Output conversion: handle both NCHW ([1,3,H,W]) and NHWC ([1,H,W,3])

  const outputImageData = new ImageData(targetSize, targetSize);

  const outputData = outputImageData.data;



  const isNCHW = outDims?.length === 4 && outDims[1] === 3; // expected

  const isNHWC = outDims?.length === 4 && outDims[3] === 3;



  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {

      const pixelIdx = y * targetSize + x;

      const dataIdx = pixelIdx * 4;



      let rOut, gOut, bOut;

      if (isNCHW) {

        rOut = output[0 * area + pixelIdx];

        gOut = output[1 * area + pixelIdx];

        bOut = output[2 * area + pixelIdx];

      } else if (isNHWC) {

        const base = pixelIdx * 3;

        rOut = output[base + 0];

        gOut = output[base + 1];

        bOut = output[base + 2];

      } else {

        // Fallback: assume NCHW

        rOut = output[pixelIdx];

        gOut = output[1 * area + pixelIdx];

        bOut = output[2 * area + pixelIdx];

      }



      // Convert from [-1, 1] to [0, 255]

      outputData[dataIdx] = Math.round(Math.max(0, Math.min(255, ((rOut + 1) / 2) * 255)));

      outputData[dataIdx + 1] = Math.round(Math.max(0, Math.min(255, ((gOut + 1) / 2) * 255)));

      outputData[dataIdx + 2] = Math.round(Math.max(0, Math.min(255, ((bOut + 1) / 2) * 255)));

      outputData[dataIdx + 3] = 255;

    }

  }



  // Canvas to hold padded output

  const outputCanvas = document.createElement('canvas');

  outputCanvas.width = targetSize;

  outputCanvas.height = targetSize;

  const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

  outputCtx.putImageData(outputImageData, 0, 0);



  // Crop back to processing size (remove padding) then resize to original crop size

  const processedCanvas = document.createElement('canvas');

  processedCanvas.width = processW;

  processedCanvas.height = processH;

  const processedCtx = processedCanvas.getContext('2d', { willReadFrequently: true });

  processedCtx.drawImage(outputCanvas, 0, 0, processW, processH, 0, 0, processW, processH);



  const inpaintedCanvas = document.createElement('canvas');

  inpaintedCanvas.width = cropW;

  inpaintedCanvas.height = cropH;

  const inpaintedCtx = inpaintedCanvas.getContext('2d', { willReadFrequently: true });

  inpaintedCtx.drawImage(processedCanvas, 0, 0, processW, processH, 0, 0, cropW, cropH);



  // Blend only the masked region back onto the original crop (soft edges like Python)

  const inpaintedData = inpaintedCtx.getImageData(0, 0, cropW, cropH);
  const inpaintedPixels = inpaintedData.data;
  const originalPixels = originalCrop.data;
  const blurredMask = blurMask(cropMask, cropW, cropH, 2);
  const blendedPixels = new Uint8ClampedArray(cropW * cropH * 4);


  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const idx = y * cropW + x;
      const base = idx * 4;
      // Ensure masked core is fully replaced; soften only at edges
      const hard = cropMask[idx] >= 128 ? 1 : 0;
      const soft = Math.min(1, blurredMask[idx] / 255);
      const alpha = hard ? 1 : soft;
      blendedPixels[base] = Math.round(inpaintedPixels[base] * alpha + originalPixels[base] * (1 - alpha));
      blendedPixels[base + 1] = Math.round(inpaintedPixels[base + 1] * alpha + originalPixels[base + 1] * (1 - alpha));
      blendedPixels[base + 2] = Math.round(inpaintedPixels[base + 2] * alpha + originalPixels[base + 2] * (1 - alpha));
      blendedPixels[base + 3] = 255;
    }
  }



  inpaintedCtx.putImageData(new ImageData(blendedPixels, cropW, cropH), 0, 0);



  // Create result canvas

  const resultCanvas = document.createElement('canvas');

  resultCanvas.width = width;

  resultCanvas.height = height;

  const resultCtx = resultCanvas.getContext('2d', { willReadFrequently: true });



  // Copy original image then overlay blended inpainted patch

  resultCtx.drawImage(imageCanvas, 0, 0);

  resultCtx.drawImage(inpaintedCanvas, minX, minY);



  console.log('[WatermarkRemover] Inpainting complete!');

  return resultCanvas;

}





// Fetch image with CORS workaround - uses background script to bypass CORS

async function fetchImageAsBlob(url) {

  console.log('[WatermarkRemover] Fetching image:', url);



  // First, try to fetch via background script (bypasses CORS)

  try {

    const blob = await new Promise((resolve, reject) => {

      chrome.runtime.sendMessage({ action: 'fetchImage', url }, response => {

        if (chrome.runtime.lastError) {

          reject(new Error(chrome.runtime.lastError.message));

        } else if (response && response.success) {

          const binary = atob(response.data);

          const bytes = new Uint8Array(binary.length);

          for (let i = 0; i < binary.length; i++) {

            bytes[i] = binary.charCodeAt(i);

          }

          resolve(new Blob([bytes], { type: response.mimeType || 'image/png' }));

        } else {

          reject(new Error(response?.error || 'Failed to fetch image via background'));

        }

      });

    });

    console.log('[WatermarkRemover] Fetched via background script');

    return blob;

  } catch (bgError) {

    console.log('[WatermarkRemover] Background fetch failed:', bgError.message);

  }



  // Fallback: try direct fetch with CORS

  try {

    const response = await fetch(url, { mode: 'cors' });

    if (response.ok) {

      console.log('[WatermarkRemover] Fetched via direct CORS request');

      return await response.blob();

    }

  } catch (e) {

    console.log('[WatermarkRemover] Direct fetch failed:', e.message);

  }



  throw new Error('Could not fetch image - all methods failed');

}



// Process a single image

async function processImage(imageUrl, options = { silent: false }) {
  console.log('[WatermarkRemover] processImage called with:', imageUrl);

  if (!imageUrl) {
    if (!options.silent) showNotification('Error: No image URL');
    throw new Error('No image URL provided');

  }



  try {

    if (!options.silent) showNotification('Loading image...');



    let canvas = null;



    // Try to find the image element on the page

    // Check both src and currentSrc, and also data-src for lazy loaded images

    let imgElement = null;

    const allImages = document.querySelectorAll('img');

    for (const img of allImages) {

      if (img.src === imageUrl || img.currentSrc === imageUrl ||

        img.getAttribute('data-src') === imageUrl) {

        imgElement = img;

        break;

      }

    }



    if (imgElement && imgElement.complete && imgElement.naturalWidth > 0) {

      console.log('[WatermarkRemover] Found image element:', imgElement.naturalWidth, 'x', imgElement.naturalHeight);



      canvas = document.createElement('canvas');

      canvas.width = imgElement.naturalWidth;

      canvas.height = imgElement.naturalHeight;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });



      try {

        ctx.drawImage(imgElement, 0, 0);

        // Test if we can read pixels (will throw if tainted)

        ctx.getImageData(0, 0, 1, 1);

        console.log('[WatermarkRemover] Successfully drew image from element');

      } catch (e) {

        console.log('[WatermarkRemover] Canvas tainted, will fetch via background script');

        canvas = null;

      }

    }



    // If canvas is null (tainted or element not found), fetch via background script

    if (!canvas) {

      console.log('[WatermarkRemover] Fetching image via background script...');

      const blob = await fetchImageAsBlob(imageUrl);

      const blobUrl = URL.createObjectURL(blob);



      const img = new Image();

      await new Promise((resolve, reject) => {

        img.onload = resolve;

        img.onerror = () => reject(new Error('Failed to load image from blob'));

        img.src = blobUrl;

      });



      canvas = document.createElement('canvas');

      canvas.width = img.width;

      canvas.height = img.height;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      ctx.drawImage(img, 0, 0);



      URL.revokeObjectURL(blobUrl);

      console.log('[WatermarkRemover] Image loaded from blob:', img.width, 'x', img.height);

    }



    if (canvas.width < 50 || canvas.height < 50) {

      throw new Error('Image too small');

    }



    if (canvas.width < 50 || canvas.height < 50) {

      throw new Error('Image too small');

    }



    if (!options.silent) showNotification('Removing watermark...');

    const result = await inpaint(canvas);

    // Check if we actually did anything (inpaint returns original canvas if no mask)
    // We can check if getContext is different or compare data, but inpaint() 
    // prints "No mask detected" and returns input canvas.
    // If we are in batch mode, we might not want to download untouched images?
    // For now, consistent behavior: always download.

    // Optionally download mask overlay for debugging alignment
    if (DEBUG_MASK_DOWNLOAD && window.__lastMaskData) {
      const { width: maskW, height: maskH, mask, imageData } = window.__lastMaskData;
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = maskW;
      maskCanvas.height = maskH;
      const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
      const overlay = maskCtx.createImageData(maskW, maskH);
      for (let i = 0; i < maskW * maskH; i++) {
        const base = i * 4;
        const m = mask[i];
        overlay.data[base] = imageData[base];
        overlay.data[base + 1] = imageData[base + 1];
        overlay.data[base + 2] = imageData[base + 2];
        overlay.data[base + 3] = 255;
        if (m > 0) {
          overlay.data[base] = Math.min(255, overlay.data[base] + 120);
        }
      }
      maskCtx.putImageData(overlay, 0, 0);
      const maskDataUrl = maskCanvas.toDataURL('image/png');
      downloadImage(maskDataUrl, 'watermark-mask-debug.png');
    }

    const dataUrl = result.toDataURL('image/png');

    // Always download the processed image
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `watermark-removed-${timestamp}.png`;
    downloadImage(dataUrl, filename);

    if (!options.silent) showNotification('✓ Image downloaded!');



    return dataUrl;



  } catch (error) {

    const errorMsg = error.message || error.toString() || 'Unknown error';

    console.error('Failed to process image:', error);

    showNotification('Failed: ' + errorMsg);

    throw error;

  }

}



// Download image helper function

function downloadImage(dataUrl, filename) {

  const link = document.createElement('a');

  link.href = dataUrl;

  link.download = filename;

  link.style.display = 'none';

  document.body.appendChild(link);

  link.click();

  document.body.removeChild(link);

  console.log('[WatermarkRemover] Downloaded:', filename);

}



// Show notification toast

function showNotification(message) {

  const existing = document.getElementById('watermark-remover-notification');

  if (existing) existing.remove();



  const notification = document.createElement('div');

  notification.id = 'watermark-remover-notification';

  notification.textContent = message;

  notification.style.cssText = `

    position: fixed;

    bottom: 20px;

    right: 20px;

    padding: 12px 24px;

    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

    color: white;

    border-radius: 8px;

    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

    font-size: 14px;

    font-weight: 500;

    z-index: 999999;

    box-shadow: 0 4px 20px rgba(0,0,0,0.3);

  `;



  document.body.appendChild(notification);

  setTimeout(() => notification.remove(), 4000);

}



// Listen for messages

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  console.log('Content script received:', message);



  if (message.action === 'processImage') {

    processImage(message.imageUrl)

      .then(() => sendResponse({ success: true }))

      .catch(err => sendResponse({ success: false, error: err.message || String(err) }));

    return true;

  }



  if (message.action === 'showNotification') {

    showNotification(message.message);

    sendResponse({ success: true });

  }



  if (message.action === 'ping') {

    sendResponse({ success: true, loaded: true });

  }



  if (message.action === 'processAllImages') {
    processAllImages()
      .then(() => sendResponse({ success: true }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // async
  }

  return false;

});

async function processAllImages() {
  // Find all images
  const images = Array.from(document.querySelectorAll('img'));

  // Filter for reasonable size (ignore tiny icons, tracking pixels)
  // We use naturalWidth if loaded, otherwise we wait? 
  // Just filter what's ready.
  const candidates = images.filter(img => {
    return img.complete && img.naturalWidth > 150 && img.naturalHeight > 150;
  });

  // Dedup URLs
  const urls = [...new Set(candidates.map(img => img.src || img.currentSrc).filter(u => u))];

  if (urls.length === 0) {
    showNotification('No suitable images found on page.');
    return;
  }

  showNotification(`Found ${urls.length} images. Starting batch...`);

  let processed = 0;
  let errors = 0;

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    showNotification(`Batch: Processing ${i + 1}/${urls.length}...`);
    try {
      await processImage(url, { silent: true }); // Silent processing
      processed++;
    } catch (e) {
      console.error(`Failed to process ${url}:`, e);
      errors++;
    }

    // Small delay to let UI breathe
    await new Promise(r => setTimeout(r, 200));
  }

  showNotification(`Batch complete! Processed: ${processed}, Failed: ${errors}`);
}



console.log('Gemini Watermark Remover loaded on:', window.location.href);

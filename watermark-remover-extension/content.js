// Content script for Star Mark Remover
// Handles image processing on web pages - Uses offscreen document for non-blocking processing

// Enable to download a visualized mask overlay for debugging
const DEBUG_MASK_DOWNLOAD = false;

// Process image using the background script's offscreen document (non-blocking)
async function processWithWorker(imageData, width, height) {
  return new Promise((resolve, reject) => {
    // Convert TypedArray to regular array for message passing
    const dataArray = Array.from(imageData);

    chrome.runtime.sendMessage({
      action: 'processInOffscreen',
      imageData: dataArray,
      width: width,
      height: height
    }, response => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else if (response.error) {
        reject(new Error(response.error));
      } else {
        resolve(response);
      }
    });
  });
}


// Check if ONNX Runtime is available (loaded via manifest)
function checkONNXRuntime() {
  if (typeof ort !== 'undefined') {
    console.log('[StarMarkRemover] ONNX Runtime is available');
    return true;
  }

  console.error('[StarMarkRemover] ONNX Runtime not loaded');

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

    showNotification('Loading Masking Model...');



    if (!checkONNXRuntime()) {
      throw new Error('ONNX Runtime not available');
    }

    // Suppress annoying "Unknown CPU vendor" warning from ONNX Runtime (harmless in browser)
    const originalLog = console.log;
    const originalWarn = console.warn;
    const filterFn = (args) => {
      const msg = args[0];
      return typeof msg === 'string' && (msg.includes('Unknown CPU vendor') || msg.includes('cpuinfo_vendor value'));
    };

    console.warn = function (...args) { if (!filterFn(args)) originalWarn.apply(console, args); };
    console.log = function (...args) { if (!filterFn(args)) originalLog.apply(console, args); };

    // Configure ONNX Runtime
    ort.env.logLevel = 'error';



    // Configure WASM paths to use extension's bundled files

    const wasmPath = chrome.runtime.getURL('lib/');

    ort.env.wasm.wasmPaths = wasmPath;

    // Use single-threaded mode to avoid cross-origin isolation requirement

    ort.env.wasm.numThreads = 1;

    console.log('[StarMarkRemover] WASM path:', wasmPath);



    const modelUrl = 'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx';



    console.log('[StarMarkRemover] Loading MI-GAN model from:', modelUrl);

    modelSession = await ort.InferenceSession.create(modelUrl, {

      executionProviders: ['webgpu', 'wasm'],

      graphOptimizationLevel: 'all'

    });



    modelInputName = modelSession.inputNames[0];

    modelOutputName = modelSession.outputNames[0];



    console.log('[StarMarkRemover] Model input name:', modelInputName);

    console.log('[StarMarkRemover] Model output name:', modelOutputName);



    isModelLoaded = true;

    isModelLoading = false;

    console.log('[StarMarkRemover] MI-GAN model loaded successfully!');

    showNotification('AI model loaded!');

  } catch (error) {

    console.error('[StarMarkRemover] Failed to load model:', error);

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



// Build integral image (summed-area table) for O(1) rectangular sum queries
function buildIntegralImage(data, w, h) {
  // Use Float64Array to avoid precision issues with large sums
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

// Query rectangular sum from integral image in O(1)
function rectSum(integral, stride, x1, y1, x2, y2) {
  // Returns sum of pixels in rect [x1,y1] to [x2,y2] inclusive
  return integral[(y2 + 1) * stride + (x2 + 1)]
    - integral[(y1) * stride + (x2 + 1)]
    - integral[(y2 + 1) * stride + (x1)]
    + integral[(y1) * stride + (x1)];
}

// Template matching using normalized cross-correlation with integral image optimization
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

      // Correlation still requires O(templateSize²) but this is unavoidable for NCC
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
function detectStarWatermark(imageData, width, height) {
  // Convert to grayscale (matching cv2.cvtColor)
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = imageData[i * 4];
    const g = imageData[i * 4 + 1];
    const b = imageData[i * 4 + 2];
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



// Blur mask for soft edge blending (box blur approximation of Gaussian)
function blurMask(mask, width, height, radius) {
  const output = new Uint8Array(width * height);
  const kernelSize = radius * 2 + 1;
  const kernelArea = kernelSize * kernelSize;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let ky = -radius; ky <= radius; ky++) {
        for (let kx = -radius; kx <= radius; kx++) {
          const ny = Math.max(0, Math.min(height - 1, y + ky));
          const nx = Math.max(0, Math.min(width - 1, x + kx));
          sum += mask[ny * width + nx];
        }
      }
      output[y * width + x] = Math.round(sum / kernelArea);
    }
  }

  return output;
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
    console.warn('[StarMarkRemover] No mask detected, returning original');
    showNotification('No star mark detected');
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



  console.log(`[StarMarkRemover] Crop region: (${minX},${minY}) to (${maxX},${maxY}), size: ${cropW}x${cropH}`);



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



  console.log('[StarMarkRemover] Running MI-GAN inference...');

  const inputTensor = new ort.Tensor('float32', inputData, [1, 4, targetSize, targetSize]);



  const feeds = {};

  feeds[modelInputName] = inputTensor;



  const results = await modelSession.run(feeds);

  const output = results[modelOutputName].data;



  const outDims = results[modelOutputName].dims;

  console.log('[StarMarkRemover] Inference complete, output shape:', outDims);



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



  console.log('[StarMarkRemover] Inpainting complete!');

  return resultCanvas;

}





// Fetch image with CORS workaround - uses background script to bypass CORS

async function fetchImageAsBlob(url) {

  console.log('[StarMarkRemover] Fetching image:', url);



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

    console.log('[StarMarkRemover] Fetched via background script');

    return blob;

  } catch (bgError) {

    console.log('[StarMarkRemover] Background fetch failed:', bgError.message);

  }



  // Fallback: try direct fetch with CORS

  try {

    const response = await fetch(url, { mode: 'cors' });

    if (response.ok) {

      console.log('[StarMarkRemover] Fetched via direct CORS request');

      return await response.blob();

    }

  } catch (e) {

    console.log('[StarMarkRemover] Direct fetch failed:', e.message);

  }



  throw new Error('Could not fetch image - all methods failed');

}



// Process a single image

async function processImage(imageUrl, options = { silent: false }) {
  console.log('[StarMarkRemover] processImage called with:', imageUrl);

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

      console.log('[StarMarkRemover] Found image element:', imgElement.naturalWidth, 'x', imgElement.naturalHeight);



      canvas = document.createElement('canvas');

      canvas.width = imgElement.naturalWidth;

      canvas.height = imgElement.naturalHeight;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });



      try {

        ctx.drawImage(imgElement, 0, 0);

        // Test if we can read pixels (will throw if tainted)

        ctx.getImageData(0, 0, 1, 1);

        console.log('[StarMarkRemover] Successfully drew image from element');

      } catch (e) {

        console.log('[StarMarkRemover] Canvas tainted, will fetch via background script');

        canvas = null;

      }

    }



    // If canvas is null (tainted or element not found), fetch via background script

    if (!canvas) {

      console.log('[StarMarkRemover] Fetching image via background script...');

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

      console.log('[StarMarkRemover] Image loaded from blob:', img.width, 'x', img.height);

    }



    if (canvas.width < 50 || canvas.height < 50) {

      throw new Error('Image too small');

    }



    if (canvas.width < 50 || canvas.height < 50) {

      throw new Error('Image too small');

    }



    // Use Web Worker for non-blocking processing
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const workerResult = await processWithWorker(
      imageData.data,
      canvas.width,
      canvas.height
    );

    // Check if no star mark was detected
    if (workerResult.noMask) {
      showNotification('No star mark detected');
      // Still return original for consistency
    }

    // Create result canvas from worker output
    const result = document.createElement('canvas');
    result.width = canvas.width;
    result.height = canvas.height;
    const resultCtx = result.getContext('2d', { willReadFrequently: true });

    // Put the processed imageData back
    const resultImageData = new ImageData(
      new Uint8ClampedArray(workerResult.imageData),
      canvas.width,
      canvas.height
    );
    resultCtx.putImageData(resultImageData, 0, 0);

    // Use user preferences for compression
    const storage = await chrome.storage.local.get(['compressImage', 'compressionQuality']);
    const useJpeg = storage.compressImage === true;
    const quality = (storage.compressionQuality || 80) / 100;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    let dataUrl, filename;

    if (useJpeg) {
      dataUrl = result.toDataURL('image/jpeg', quality);
      filename = `star-mark-removed-${timestamp}.jpg`;
    } else {
      dataUrl = result.toDataURL('image/png');
      filename = `star-mark-removed-${timestamp}.png`;
    }

    downloadImage(dataUrl, filename);

    if (!options.silent) showNotification('Image downloaded successfully');



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

  console.log('[StarMarkRemover] Downloaded:', filename);

}



// Show notification toast

function showNotification(message) {
  const existing = document.getElementById('watermark-remover-notification');
  if (existing) existing.remove();

  // Determine if success message for green tick
  const isSuccess = message.toLowerCase().includes('successfully') || message.toLowerCase().includes('done') || message.toLowerCase().includes('complete');

  const notification = document.createElement('div');
  notification.id = 'watermark-remover-notification';

  // SVG Checkmark
  const iconHtml = isSuccess ?
    `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 12px; flex-shrink: 0;">
        <path d="M12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2C6.5 2 2 6.5 2 12C2 17.5 6.5 22 12 22Z" fill="#22c55e" fill-opacity="0.2"/>
        <path d="M7.75 11.9999L10.58 14.8299L16.25 9.16992" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
     </svg>` : '';

  notification.innerHTML = `<div style="display:flex; align-items:center;">${iconHtml}<span>${message}</span></div>`;

  // Modern, Minimalist "Vercel-like" Design
  notification.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 12px 16px;
    background: #171717; /* Solid dark grey/black */
    color: #ededed;
    border: 1px solid #333333;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    letter-spacing: -0.01em;
    font-weight: 500;
    z-index: 2147483647; /* Max Z-index */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  `;

  document.body.appendChild(notification);

  // Animate in
  requestAnimationFrame(() => {
    notification.style.opacity = '1';
    notification.style.transform = 'translateY(0)';
  });

  // Animate out
  setTimeout(() => {
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(4px)';
    setTimeout(() => notification.remove(), 200);
  }, 4000);
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

console.log('Star Mark Remover loaded on:', window.location.href);

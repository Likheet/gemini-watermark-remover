// Dashboard script for Star Mark Remover - 3-Page Redesign
// Uses Web Worker for non-blocking processing

// --- Elements ---
const navBtns = document.querySelectorAll('.nav-btn');
const pages = document.querySelectorAll('.page-container');

// Single Elements
const dropZoneSingle = document.getElementById('dropZoneSingle');
const fileInputSingle = document.getElementById('fileInputSingle');
const previewImageSingle = document.getElementById('previewImageSingle');
const singlePlaceholder = document.getElementById('singlePlaceholder');
const processBtnSingle = document.getElementById('processBtnSingle');
const clearBtnSingle = document.getElementById('clearBtnSingle');
const resultCanvasSingle = document.getElementById('resultCanvasSingle');
const singleResultPlaceholder = document.getElementById('singleResultPlaceholder');
const downloadBtnSingle = document.getElementById('downloadBtnSingle');

// Batch Elements
const dropZoneBatch = document.getElementById('dropZoneBatch');
const fileInputBatch = document.getElementById('fileInputBatch');
const processBtnBatch = document.getElementById('processBtnBatch');
const clearBtnBatch = document.getElementById('clearBtnBatch');
const queueContainer = document.getElementById('queueContainer');
const batchActions = document.getElementById('batchActions');
const downloadAllBtn = document.getElementById('downloadAllBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');

// Manual Elements
const dropZoneManual = document.getElementById('dropZoneManual');
const fileInputManual = document.getElementById('fileInputManual');
const manualCanvas = document.getElementById('manualCanvas');
const manualCanvasWrapper = document.getElementById('manualCanvasWrapper');
const manualPlaceholder = document.getElementById('manualPlaceholder');
const brushSizeInput = document.getElementById('brushSize');
const clearMaskBtn = document.getElementById('clearMaskBtn');
const undoMaskBtn = document.getElementById('undoMaskBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');
const processBtnManual = document.getElementById('processBtnManual');
const resetManualBtn = document.getElementById('resetManualBtn');
const manualResultImg = document.getElementById('manualResultImg');
const manualResultPlaceholder = document.getElementById('manualResultPlaceholder');
const downloadBtnManual = document.getElementById('downloadBtnManual');

// Common
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const progressContainer = document.getElementById('progressContainer');
const progressLine = document.getElementById('progressLine');

// --- State ---
let currentPage = 'single';
let worker = null;
let isProcessing = false; // Global processing lock

// Single State
let singleFile = null;

// Batch State
let queue = [];
let isBatchProcessing = false;
let currentBatchId = null;

// Manual State
let manualFile = null;
let manualImgHelper = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let manualCtx = null;
let manualMaskCtx = null;
let maskCanvas = null;
let strokeHistory = [];      // Stores canvas states for undo
let maskStrokeHistory = [];  // Stores mask states for undo

// --- Notification Sound ---
function playNotificationSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        // Pleasant "ding" sound - two quick tones
        oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
        oscillator.frequency.setValueAtTime(1320, audioCtx.currentTime + 0.1); // E6 note

        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

        oscillator.start(audioCtx.currentTime);
        oscillator.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        console.log('Audio notification not supported');
    }
}

// --- Initialization ---

function init() {
    initWorker();
    setupNav();
    setupSinglePage();
    setupBatchPage();
    setupManualPage();
    setupWelcomeModal();
}

function updateStatus(status, message) {
    statusText.textContent = message;
    statusDot.className = 'status-dot';

    if (status === 'ready') {
        statusDot.classList.add('active');
        showProgress(false);
        setProcessingLock(false);
    } else if (status === 'processing' || status === 'loading') {
        statusDot.classList.add('processing');
        if (status === 'processing') setProcessingLock(true);
    } else if (status === 'error') {
        statusDot.style.background = '#ef4444';
        showProgress(false);
        setProcessingLock(false);
    }
}

function setProcessingLock(locked) {
    isProcessing = locked;
    navBtns.forEach(btn => {
        if (locked) {
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
            btn.title = "Currently Working...";
        } else {
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
            btn.title = "";
        }
    });
}

function showProgress(show, percent = 0) {
    if (show) {
        progressContainer.classList.add('active');

        // Single Page Placeholder Progress
        if (currentPage === 'single') {
            const orb = document.getElementById('singleProcessingOrb');
            const text = document.querySelector('#singleResultPlaceholder .placeholder-text');
            if (orb) orb.parentElement.classList.add('loading');
            if (text) {
                text.textContent = "Generating";
                text.classList.add('breathing-text');
            }
        }

        // Manual Page Placeholder Progress
        if (currentPage === 'manual') {
            const orb = document.getElementById('manualProcessingOrb');
            const text = document.querySelector('#manualResultPlaceholder .placeholder-text');
            if (orb) orb.parentElement.classList.add('loading');
            if (text) {
                text.textContent = "Generating";
                text.classList.add('breathing-text');
            }
        }
    } else {
        progressContainer.classList.remove('active');

        // Reset Single
        const singleOrb = document.getElementById('singleProcessingOrb');
        const singleText = document.querySelector('#singleResultPlaceholder .placeholder-text');
        if (singleOrb) singleOrb.parentElement.classList.remove('loading');
        if (singleText) {
            singleText.textContent = "Result will appear here";
            singleText.classList.remove('breathing-text');
        }

        // Reset Manual
        const manualOrb = document.getElementById('manualProcessingOrb');
        const manualText = document.querySelector('#manualResultPlaceholder .placeholder-text');
        if (manualOrb) manualOrb.parentElement.classList.remove('loading');
        if (manualText) {
            manualText.textContent = "Result will appear here";
            manualText.classList.remove('breathing-text');
        }
    }
}

function initWorker() {
    updateStatus('loading', 'Initializing Masking Model...');
    worker = new Worker('processor.worker.js');
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (err) => {
        console.error('Worker error:', err);
        updateStatus('error', 'Worker failed');
        isBatchProcessing = false;
        setProcessingLock(false);
    };
    const wasmPath = chrome.runtime.getURL('lib/');
    worker.postMessage({ type: 'init', config: { wasmPath } });
}

function handleWorkerMessage(e) {
    const { type, status, message, percent, imageData, width, height, noMask } = e.data;

    if (type === 'status') {
        updateStatus(status, message);
    } else if (type === 'progress') {
        showProgress(true, percent);
    } else if (type === 'result') {
        // Worker sends raw imageData buffer, width, height - need to create ImageBitmap
        if (imageData && width && height) {
            const imgData = new ImageData(new Uint8ClampedArray(imageData), width, height);
            createImageBitmap(imgData).then(bitmap => {
                handleResult(bitmap);
            }).catch(err => {
                console.error('Failed to create ImageBitmap:', err);
                updateStatus('error', 'Processing failed');
            });
        } else {
            console.error('Invalid result data from worker');
            updateStatus('error', 'Processing failed - invalid result');
        }
    } else if (type === 'error') {
        console.error('Worker error:', message);
        updateStatus('error', 'Error: ' + message);
        if (currentPage === 'batch') handleBatchError(message);
    }
}

// --- Navigation ---

function setupNav() {
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (isProcessing) {
                statusText.textContent = "Please wait, currently working...";
                setTimeout(() => { if (isProcessing) statusText.textContent = "Processing..."; }, 2000);
                return;
            }

            navBtns.forEach(b => b.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const pageId = btn.dataset.page;

            const elementId = 'page' + pageId.charAt(0).toUpperCase() + pageId.slice(1);
            const pageEl = document.getElementById(elementId);
            if (pageEl) {
                pageEl.classList.add('active');
                currentPage = pageId;
            }
        });
    });
}

// --- Single Page Logic ---

function setupSinglePage() {
    setupDragDrop(dropZoneSingle, fileInputSingle, handleSingleFile);

    clearBtnSingle.addEventListener('click', () => {
        singleFile = null;
        previewImageSingle.src = '';
        previewImageSingle.classList.add('hidden');
        singlePlaceholder.classList.remove('hidden');
        processBtnSingle.disabled = true;

        resultCanvasSingle.classList.add('hidden');
        singleResultPlaceholder.classList.remove('hidden');
        downloadBtnSingle.disabled = true;

        fileInputSingle.value = '';
    });

    processBtnSingle.addEventListener('click', () => {
        if (!singleFile) return;
        processBtnSingle.disabled = true;
        sendProcessRequest(singleFile, null, 'single');
    });

    downloadBtnSingle.addEventListener('click', () => {
        downloadCanvas(resultCanvasSingle, `cleaned-${singleFile?.name || 'image'}`);
    });
}

function handleSingleFile(file) {
    if (!file.type.startsWith('image/')) return;
    singleFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImageSingle.src = e.target.result;
        previewImageSingle.classList.remove('hidden');
        singlePlaceholder.classList.add('hidden');
        processBtnSingle.disabled = false;

        resultCanvasSingle.classList.add('hidden');
        singleResultPlaceholder.classList.remove('hidden');
        downloadBtnSingle.disabled = true;
    };
    reader.readAsDataURL(file);
}

// --- Batch Page Logic ---

function setupBatchPage() {
    // Make sure the picker truly allows multiple files (some browsers can ignore the attribute)
    fileInputBatch.multiple = true;
    fileInputBatch.setAttribute('multiple', 'multiple');

    // Ensure the handler always receives an array of files
    setupDragDrop(dropZoneBatch, fileInputBatch, handleBatchFiles, true);  // true = batch mode
    dropZoneBatch.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZoneBatch.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files || []);
        if (files.length) handleBatchFiles(files);
    });
    // Note: change listener is already added by setupDragDrop

    processBtnBatch.addEventListener('click', processBatchQueue);
    clearBtnBatch.addEventListener('click', () => {
        if (isBatchProcessing) return;
        queue = [];
        renderQueue();
        updateBatchUI();
    });

    downloadAllBtn.addEventListener('click', downloadAllBatch);
    downloadZipBtn.addEventListener('click', downloadBatchZip);
}

function handleBatchFiles(fileList) {
    const files = Array.isArray(fileList) ? fileList : Array.from(fileList || []);
    if (!files.length) return;

    let addedCount = 0;
    files.forEach((file, index) => { // Use index for unique ID
        if (!file.type.startsWith('image/')) return;

        // Ensure strictly unique ID by combining time + random + index
        const id = Date.now() + '-' + index + '-' + Math.random().toString(36).substr(2, 9);
        const item = { id, file, name: file.name, status: 'pending', resultDataUrl: null, thumbUrl: null };
        queue.push(item);
        addedCount++;

        const reader = new FileReader();
        reader.onload = (e) => {
            item.thumbUrl = e.target.result;
            renderQueue();
        };
        reader.readAsDataURL(file);
    });

    renderQueue();
    updateBatchUI();

    if (fileInputBatch) fileInputBatch.value = '';
}

function renderQueue() {
    queueContainer.innerHTML = '';
    queue.forEach(item => {
        const div = document.createElement('div');
        div.className = `queue-item ${item.status}`;
        div.innerHTML = `
            <img class="queue-thumb" src="${item.thumbUrl || ''}">
            <div class="queue-info">
                <div class="queue-name">${item.name}</div>
                <div class="queue-status">${getStatusIcon(item.status)} ${item.status}</div>
            </div>
            <div class="queue-actions"></div>
        `;

        const actions = div.querySelector('.queue-actions');
        if (item.status === 'done' && item.resultDataUrl) {
            const dlBtn = document.createElement('button');
            dlBtn.className = 'primary btn-text';
            dlBtn.style.background = 'var(--primary)'; dlBtn.style.color = 'var(--primary-fg)';
            dlBtn.textContent = 'Download';
            dlBtn.onclick = () => downloadDataUrl(item.resultDataUrl, item.name);
            actions.appendChild(dlBtn);
        }
        if (item.status !== 'processing') {
            const rmBtn = document.createElement('button');
            rmBtn.className = 'btn-text';
            rmBtn.innerHTML = '✕';
            rmBtn.onclick = () => { queue = queue.filter(x => x.id !== item.id); renderQueue(); updateBatchUI(); };
            actions.appendChild(rmBtn);
        }
        queueContainer.appendChild(div);
    });
}

function updateBatchUI() {
    processBtnBatch.disabled = isBatchProcessing || queue.filter(i => i.status === 'pending').length === 0;

    // Show download buttons section when at least 1 image is done
    const anyDone = queue.some(i => i.status === 'done');
    const allDone = queue.length > 0 && queue.every(i => i.status === 'done');

    if (anyDone) {
        batchActions.classList.remove('hidden');
        // Enable buttons only when there's at least one done
        downloadAllBtn.disabled = false;
        downloadZipBtn.disabled = false;
    } else {
        batchActions.classList.add('hidden');
        downloadAllBtn.disabled = true;
        downloadZipBtn.disabled = true;
    }
}

function downloadAllBatch() {
    chrome.storage.local.get(['compressImage', 'compressionQuality'], (res) => {
        const quality = (res.compressionQuality || 80) / 100;
        const useJpeg = res.compressImage;
        const ext = useJpeg ? 'jpg' : 'png';

        queue.forEach((item, index) => {
            if (item.status === 'done' && item.resultDataUrl) {
                setTimeout(() => {
                    downloadDataUrl(item.resultDataUrl, item.name);
                }, index * 300);
            }
        });
    });
}

function downloadBatchZip() {
    const zip = new JSZip();
    const folder = zip.folder("cleaned_images");

    updateStatus('processing', 'Creating ZIP...');

    chrome.storage.local.get(['compressImage', 'compressionQuality'], (res) => {
        const quality = (res.compressionQuality || 80) / 100;
        const useJpeg = res.compressImage;
        const ext = useJpeg ? '.jpg' : '.png';

        const getBlob = (dataUrl) => new Promise(resolve => {
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = img.width; c.height = img.height;
                c.getContext('2d').drawImage(img, 0, 0);
                c.toBlob(blob => resolve(blob), useJpeg ? 'image/jpeg' : 'image/png', quality);
            };
            img.src = dataUrl;
        });

        const promises = queue.map(async item => {
            if (item.status === 'done' && item.resultDataUrl) {
                const blob = await getBlob(item.resultDataUrl);
                folder.file(item.name.replace(/\.[^.]+$/, '') + ext, blob);
            }
        });

        Promise.all(promises).then(() => {
            zip.generateAsync({ type: "blob" }).then(function (content) {
                const link = document.createElement('a');
                link.href = URL.createObjectURL(content);
                link.download = "cleaned_images.zip";
                link.click();
                updateStatus('ready', 'ZIP Downloaded!');
            });
        });
    });
}

// --- Manual Page Logic ---

function setupManualPage() {
    setupDragDrop(dropZoneManual, fileInputManual, handleManualFile);

    // Manual Warning Banner Logic
    const manualWarningBanner = document.getElementById('manualWarningBanner');
    const closeManualWarning = document.getElementById('closeManualWarning');

    if (!localStorage.getItem('manualWarningDismissed')) {
        manualWarningBanner.classList.remove('hidden');
    }

    closeManualWarning.addEventListener('click', () => {
        manualWarningBanner.classList.add('hidden');
        localStorage.setItem('manualWarningDismissed', 'true');
    });

    manualCtx = manualCanvas.getContext('2d', { willReadFrequently: true });

    // Brush cursor element
    const brushCursor = document.getElementById('brushCursor');

    manualCanvas.addEventListener('mousedown', startDrawing);
    manualCanvas.addEventListener('mousemove', (e) => {
        draw(e);
        if (brushCursor && manualImgHelper) {
            const wrapperRect = manualCanvasWrapper.getBoundingClientRect();
            // Cursor position relative to wrapper top-left
            brushCursor.style.left = (e.clientX - wrapperRect.left) + 'px';
            brushCursor.style.top = (e.clientY - wrapperRect.top) + 'px';
        }
    });
    manualCanvas.addEventListener('mouseup', stopDrawing);
    manualCanvas.addEventListener('mouseout', () => {
        stopDrawing();
        if (brushCursor) brushCursor.style.display = 'none';
    });
    manualCanvas.addEventListener('mouseenter', () => {
        if (brushCursor && manualImgHelper) brushCursor.style.display = 'block';
    });

    // Update cursor size
    brushSizeInput.addEventListener('input', () => {
        if (brushCursor && manualImgHelper) {
            const clientWidth = manualCanvas.clientWidth;
            const ratio = clientWidth / manualCanvas.width;
            const size = parseInt(brushSizeInput.value) * ratio;
            brushCursor.style.width = size + 'px';
            brushCursor.style.height = size + 'px';
        }
    });

    undoMaskBtn.addEventListener('click', () => {
        if (strokeHistory.length === 0 || !manualImgHelper) return;
        const prevCanvas = strokeHistory.pop();
        const prevMask = maskStrokeHistory.pop();
        manualCtx.putImageData(prevCanvas, 0, 0);
        if (manualMaskCtx && prevMask) {
            manualMaskCtx.putImageData(prevMask, 0, 0);
        }
    });

    fullscreenBtn.addEventListener('click', () => {
        const wrapper = manualCanvasWrapper;
        if (!document.fullscreenElement) {
            wrapper.requestFullscreen().catch(err => console.log(err));
            fullscreenBtn.textContent = 'Exit Fullscreen';
        } else {
            document.exitFullscreen();
            fullscreenBtn.textContent = 'Fullscreen';
        }
    });

    document.addEventListener('fullscreenchange', () => {
        if (!document.fullscreenElement) fullscreenBtn.textContent = 'Fullscreen';
    });

    clearMaskBtn.addEventListener('click', () => {
        if (!manualImgHelper) return;
        strokeHistory.push(manualCtx.getImageData(0, 0, manualCanvas.width, manualCanvas.height));
        maskStrokeHistory.push(manualMaskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
        renderManualCanvas();
        if (manualMaskCtx) manualMaskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    });

    resetManualBtn.addEventListener('click', () => {
        manualFile = null;
        manualImgHelper = null;
        manualCanvas.width = 300; manualCanvas.height = 150;
        manualCtx.clearRect(0, 0, 300, 150);
        strokeHistory = [];
        maskStrokeHistory = [];
        manualCanvasWrapper.classList.add('hidden');
        manualPlaceholder.classList.remove('hidden');
        processBtnManual.disabled = true;
        manualResultImg.classList.add('hidden');
        manualResultPlaceholder.classList.remove('hidden');
        downloadBtnManual.disabled = true;
    });

    processBtnManual.addEventListener('click', () => {
        if (!manualFile || !maskCanvas) return;
        processBtnManual.disabled = true;
        const maskData = manualMaskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        sendProcessRequest(manualFile, maskData.data, 'manual');
    });

    downloadBtnManual.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = 'cleaned-manual.jpg';
        link.href = manualResultImg.src;
        link.click();
    });
}

function handleManualFile(file) {
    if (!file.type.startsWith('image/')) return;
    manualFile = file;

    const img = new Image();
    img.onload = () => {
        manualImgHelper = img;
        manualCanvasWrapper.classList.remove('hidden');
        manualPlaceholder.classList.add('hidden');
        processBtnManual.disabled = false;
        manualResultImg.classList.add('hidden');
        manualResultPlaceholder.classList.remove('hidden');
        downloadBtnManual.disabled = true;
        manualCanvas.width = img.width;
        manualCanvas.height = img.height;
        maskCanvas = document.createElement('canvas');
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;
        manualMaskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
        renderManualCanvas();
    };
    img.src = URL.createObjectURL(file);
}

function renderManualCanvas() {
    if (!manualImgHelper) return;
    manualCtx.drawImage(manualImgHelper, 0, 0);
}

function getPointerPos(e) {
    // Use offsetX/Y for coordinate relative to element padding box
    if (typeof e.offsetX === 'number') {
        const ratioX = manualCanvas.width / manualCanvas.clientWidth;
        const ratioY = manualCanvas.height / manualCanvas.clientHeight;
        return {
            x: e.offsetX * ratioX,
            y: e.offsetY * ratioY
        };
    }
    // Fallback
    const rect = manualCanvas.getBoundingClientRect();
    const scaleX = manualCanvas.width / rect.width;
    const scaleY = manualCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.stopPropagation();
    if (!manualImgHelper) return;
    strokeHistory.push(manualCtx.getImageData(0, 0, manualCanvas.width, manualCanvas.height));
    maskStrokeHistory.push(manualMaskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
    if (strokeHistory.length > 20) {
        strokeHistory.shift();
        maskStrokeHistory.shift();
    }
    isDrawing = true;
    const { x, y } = getPointerPos(e);
    lastX = x; lastY = y;
}

function draw(e) {
    if (!isDrawing || !manualImgHelper) return;
    const { x, y } = getPointerPos(e);
    const size = parseInt(brushSizeInput.value);

    manualCtx.beginPath();
    manualCtx.moveTo(lastX, lastY);
    manualCtx.lineTo(x, y);
    manualCtx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    manualCtx.lineWidth = size;
    manualCtx.lineCap = 'round';
    manualCtx.stroke();

    manualMaskCtx.beginPath();
    manualMaskCtx.moveTo(lastX, lastY);
    manualMaskCtx.lineTo(x, y);
    manualMaskCtx.strokeStyle = '#ffffff';
    manualMaskCtx.lineWidth = size;
    manualMaskCtx.lineCap = 'round';
    manualMaskCtx.stroke();
    lastX = x; lastY = y;
}

function stopDrawing() { isDrawing = false; }

// --- Generic Processing ---

async function sendProcessRequest(file, maskDataOrNull, mode) {
    updateStatus('processing', 'Loading image...');
    try {
        const imageDataObj = await getImageData(file);
        const payload = {
            type: 'process',
            data: {
                imageData: imageDataObj.data,
                width: imageDataObj.width,
                height: imageDataObj.height,
                userMask: maskDataOrNull ? maskDataOrNull : undefined
            }
        };
        const transfer = [imageDataObj.data.buffer];
        if (maskDataOrNull) transfer.push(maskDataOrNull.buffer);
        worker.postMessage(payload, transfer);
    } catch (e) {
        console.error(e);
        updateStatus('error', 'Processing failed');
        if (mode === 'batch') handleBatchError(e.message);
        else if (mode === 'single') processBtnSingle.disabled = false;
        else if (mode === 'manual') processBtnManual.disabled = false;
    }
}

// --- Batch Processing Handlers ---

function processBatchQueue() {
    if (isBatchProcessing) return;
    const nextItem = queue.find(i => i.status === 'pending');
    if (!nextItem) {
        updateStatus('ready', 'Batch Complete!');
        updateBatchUI();
        // Play notification sound when batch is complete
        playNotificationSound();
        return;
    }
    isBatchProcessing = true;
    currentBatchId = nextItem.id;
    nextItem.status = 'processing';
    renderQueue();
    updateBatchUI();
    sendProcessRequest(nextItem.file, null, 'batch');
}

function handleBatchError(msg) {
    const item = queue.find(i => i.id === currentBatchId);
    if (item) item.status = 'error';
    isBatchProcessing = false;
    renderQueue();
    setTimeout(processBatchQueue, 500);
}

function handleResult(bitmap) {
    // Defensive check: ensure bitmap is valid before processing
    if (!bitmap || typeof bitmap.width !== 'number') {
        console.error('handleResult: Invalid bitmap received', bitmap);
        updateStatus('error', 'Processing failed - invalid result');
        if (currentPage === 'batch') {
            handleBatchError('Invalid result received');
        } else if (currentPage === 'single') {
            processBtnSingle.disabled = false;
        } else if (currentPage === 'manual') {
            processBtnManual.disabled = false;
        }
        return;
    }

    if (currentPage === 'single') {
        resultCanvasSingle.width = bitmap.width;
        resultCanvasSingle.height = bitmap.height;
        resultCanvasSingle.getContext('2d').drawImage(bitmap, 0, 0);
        resultCanvasSingle.classList.remove('hidden');
        singleResultPlaceholder.classList.add('hidden');
        processBtnSingle.disabled = false;
        downloadBtnSingle.disabled = false;
        bitmap.close();
        updateStatus('ready', 'Done!');
    } else if (currentPage === 'batch') {
        const cvs = document.createElement('canvas');
        cvs.width = bitmap.width; cvs.height = bitmap.height;
        cvs.getContext('2d').drawImage(bitmap, 0, 0);
        bitmap.close();
        const item = queue.find(i => i.id === currentBatchId);
        if (item) {
            item.status = 'done';
            item.resultDataUrl = cvs.toDataURL();
        }
        isBatchProcessing = false;
        renderQueue();
        processBatchQueue();
    } else if (currentPage === 'manual') {
        const cvs = document.createElement('canvas');
        cvs.width = bitmap.width; cvs.height = bitmap.height;
        cvs.getContext('2d').drawImage(bitmap, 0, 0);
        bitmap.close();
        manualResultImg.src = cvs.toDataURL();
        manualResultImg.classList.remove('hidden');
        manualResultPlaceholder.classList.add('hidden');
        processBtnManual.disabled = false;
        downloadBtnManual.disabled = false;
        updateStatus('ready', 'Object Removed!');
    }
}

// --- Helpers ---

function setupDragDrop(zone, input, handler, isBatch = false) {
    const isMulti = isBatch || input.multiple;
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files || []);
        if (files.length) {
            if (isMulti) handler(files);
            else handler(files[0]);
        }
    });
zone.addEventListener('click', (e) => {
    const inUploadCard = e.target.closest('.upload-card');
    const inBatchBanner = e.target.closest('.batch-drop-banner');
    const isNotButton = e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT';
    const isNotCanvas = !e.target.closest('#manualCanvasWrapper') && !e.target.closest('#manualCanvas');

    if ((inUploadCard || inBatchBanner) && isNotButton && isNotCanvas) {
        input.click();
    }
});
input.addEventListener('change', (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length) {
        if (isMulti) handler(files);
        else handler(files[0]);
    }
    input.value = '';
});
}

function getStatusIcon(status) {
    if (status === 'pending') return '⏳';
    if (status === 'processing') return '🔄';
    if (status === 'done') return '✅';
    if (status === 'error') return '❌';
    return '';
}

function getImageData(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const data = ctx.getImageData(0, 0, img.width, img.height);
            resolve({ data: data.data, width: img.width, height: img.height });
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function downloadCanvas(canvas, filename) {
    chrome.storage.local.get(['compressImage', 'compressionQuality'], (res) => {
        const quality = (res.compressionQuality || 80) / 100;
        const useJpeg = res.compressImage;

        const link = document.createElement('a');
        link.download = filename + (useJpeg ? '.jpg' : '.png');
        link.href = canvas.toDataURL(useJpeg ? 'image/jpeg' : 'image/png', quality);
        link.click();
        updateStatus('ready', 'Image Downloaded!');
    });
}

function downloadDataUrl(dataUrl, originalName) {
    const link = document.createElement('a');
    // dataUrl is already formatted by the create logic (png or jpeg)
    // We just need to ensure extension matches
    const isJpeg = dataUrl.startsWith('data:image/jpeg');
    const ext = isJpeg ? '.jpg' : '.png';
    const name = originalName.replace(/\.[^.]+$/, '') + '-cleaned' + ext;

    link.href = dataUrl;
    link.download = name;
    link.click();
}

function setupWelcomeModal() {
    const modal = document.getElementById('welcomeModal');
    const closeBtn = document.getElementById('closeModalBtn');
    const getStartedBtn = document.getElementById('getStartedBtn');
    const checkbox = document.getElementById('dontShowAgainCheckbox');

    // Check storage, default is show
    chrome.storage.local.get(['hideWelcomeModal'], (res) => {
        if (!res.hideWelcomeModal) {
            modal.classList.remove('hidden');
        }
    });

    function closeModal() {
        if (checkbox.checked) {
            chrome.storage.local.set({ hideWelcomeModal: true });
        }
        modal.classList.add('hidden');
    }

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (getStartedBtn) getStartedBtn.addEventListener('click', closeModal);

    // Close on overlay click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
}

// Start
init();

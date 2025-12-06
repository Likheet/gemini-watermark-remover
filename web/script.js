// Dashboard script for Watermark Remover - Web Version
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

// --- Initialization ---

function init() {
    initWorker();
    setupNav();
    setupSinglePage();
    setupBatchPage();
    setupManualPage();
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
        const orb = document.getElementById('singleProcessingOrb');
        const text = document.querySelector('#singleResultPlaceholder .placeholder-text');

        if (currentPage === 'single' && orb) {
            orb.parentElement.classList.add('loading');
            if (text) {
                text.textContent = "Generating";
                text.classList.add('breathing-text');
            }
        }
    } else {
        progressContainer.classList.remove('active');

        const orb = document.getElementById('singleProcessingOrb');
        const text = document.querySelector('#singleResultPlaceholder .placeholder-text');

        if (orb) {
            orb.parentElement.classList.remove('loading');
            if (text) {
                text.textContent = "Result will appear here";
                text.classList.remove('breathing-text');
            }
        }
    }
}

function initWorker() {
    updateStatus('loading', 'Initializing AI Model...');
    worker = new Worker('processor.worker.js');
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (err) => {
        console.error('Worker error:', err);
        updateStatus('error', 'Worker failed');
        isBatchProcessing = false;
        setProcessingLock(false);
    };

    // Web: Use relative path
    const wasmPath = 'lib/';
    worker.postMessage({ type: 'init', config: { wasmPath } });
}

function handleWorkerMessage(e) {
    const { type, status, message, result, percent } = e.data;

    if (type === 'status') {
        updateStatus(status, message);
    } else if (type === 'progress') {
        showProgress(true, percent);
    } else if (type === 'result') {
        handleResult(result);
    } else if (type === 'error') {
        console.error('Worker error:', message);
        updateStatus('error', 'Error: ' + message);
        if (currentPage === 'batch') handleBatchError(message);
    }
}

// Helper for Web Storage
function getSettings(cb) {
    const quality = localStorage.getItem('compressionQuality') || 80;
    const useJpeg = localStorage.getItem('compressImage') === 'true';
    cb({ compressionQuality: parseInt(quality), compressImage: useJpeg });
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
    setupDragDrop(dropZoneBatch, fileInputBatch, (file) => handleBatchFiles([file]));
    dropZoneBatch.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZoneBatch.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleBatchFiles(e.dataTransfer.files);
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
    for (const file of fileList) {
        if (!file.type.startsWith('image/')) continue;
        const id = Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const item = { id, file, name: file.name, status: 'pending', resultDataUrl: null, thumbUrl: null };
        queue.push(item);

        const reader = new FileReader();
        reader.onload = (e) => { item.thumbUrl = e.target.result; renderQueue(); };
        reader.readAsDataURL(file);
    }
    renderQueue();
    updateBatchUI();
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

    const allDone = queue.length > 0 && queue.every(i => i.status === 'done');
    if (allDone) {
        batchActions.classList.remove('hidden');
    } else {
        batchActions.classList.add('hidden');
    }
}

function downloadAllBatch() {
    getSettings((res) => {
        queue.forEach((item, index) => {
            if (item.status === 'done' && item.resultDataUrl) {
                setTimeout(() => {
                    downloadDataUrl(item.resultDataUrl, item.name);
                }, index * 200);
            }
        });
    });
}

function downloadBatchZip() {
    const zip = new JSZip();
    const folder = zip.folder("cleaned_images");

    updateStatus('processing', 'Creating ZIP...');

    getSettings((res) => {
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

    manualCtx = manualCanvas.getContext('2d');

    manualCanvas.addEventListener('mousedown', startDrawing);
    manualCanvas.addEventListener('mousemove', draw);
    manualCanvas.addEventListener('mouseup', stopDrawing);
    manualCanvas.addEventListener('mouseout', stopDrawing);

    clearMaskBtn.addEventListener('click', () => {
        if (!manualImgHelper) return;
        renderManualCanvas();
        if (manualMaskCtx) manualMaskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    });

    resetManualBtn.addEventListener('click', () => {
        manualFile = null;
        manualImgHelper = null;
        manualCanvas.width = 300; manualCanvas.height = 150;
        manualCtx.clearRect(0, 0, 300, 150);

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
        manualMaskCtx = maskCanvas.getContext('2d');

        renderManualCanvas();
    };
    img.src = URL.createObjectURL(file);
}

function renderManualCanvas() {
    if (!manualImgHelper) return;
    manualCtx.drawImage(manualImgHelper, 0, 0);
}

function getPointerPos(e) {
    const rect = manualCanvas.getBoundingClientRect();
    const scaleX = manualCanvas.width / rect.width;
    const scaleY = manualCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.stopPropagation(); // Prevent opening file picker
    if (!manualImgHelper) return;
    isDrawing = true;
    const { x, y } = getPointerPos(e);
    lastX = x;
    lastY = y;
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

    lastX = x;
    lastY = y;
}

function stopDrawing() {
    isDrawing = false;
}

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
    if (!nextItem) { updateStatus('ready', 'Batch Complete!'); return; }

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
    setTimeout(processBatchQueue, 500); // Continue
}

function handleResult(bitmap) {
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
            item.resultDataUrl = cvs.toDataURL('image/png');
        }
        isBatchProcessing = false;
        renderQueue();
        setTimeout(processBatchQueue, 100);
    } else if (currentPage === 'manual') {
        const cvs = document.createElement('canvas');
        cvs.width = bitmap.width; cvs.height = bitmap.height;
        cvs.getContext('2d').drawImage(bitmap, 0, 0);
        bitmap.close();

        manualResultImg.src = cvs.toDataURL('image/jpeg');
        manualResultImg.classList.remove('hidden');
        manualResultPlaceholder.classList.add('hidden');

        processBtnManual.disabled = false;
        downloadBtnManual.disabled = false;
        updateStatus('ready', 'Mask Removed!');
    }
}

function setupDragDrop(zone, input, handler) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handler(e.dataTransfer.files[0]);
    });
    zone.addEventListener('click', (e) => {
        // Allow click if within upload-card OR batch-drop-banner
        const inUploadCard = e.target.closest('.upload-card');
        const inBatchBanner = e.target.closest('.batch-drop-banner');
        if ((inUploadCard || inBatchBanner) && e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT' && !e.target.closest('#manualCanvasWrapper') && !e.target.closest('#manualCanvas')) {
            input.click();
        }
    });
    input.addEventListener('change', (e) => {
        if (e.target.files.length) handler(e.target.files[0]);
    });
}

function getImageData(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            resolve({ data: ctx.getImageData(0, 0, c.width, c.height).data, width: c.width, height: c.height });
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function getStatusIcon(s) { return s === 'pending' ? '⏳' : s === 'processing' ? '⚙️' : s === 'done' ? '✅' : '❌'; }

function downloadCanvas(canvas, filenameRoot) {
    getSettings((res) => {
        const link = document.createElement('a');
        if (res.compressImage) {
            link.download = filenameRoot + '.jpg';
            link.href = canvas.toDataURL('image/jpeg', (res.compressionQuality || 80) / 100);
        } else {
            link.download = filenameRoot + '.png';
            link.href = canvas.toDataURL('image/png');
        }
        link.click();
    });
}

function downloadDataUrl(dataUrl, filenameRoot) {
    const link = document.createElement('a');
    link.download = filenameRoot + '.png';
    link.href = dataUrl;
    link.click();
}

// Start
init();

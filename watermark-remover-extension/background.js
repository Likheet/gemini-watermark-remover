// Background service worker for Star Mark Remover

// Initialize context menu
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'removeWatermark',
    title: 'Remove Star Mark',
    contexts: ['image']
  });

  chrome.contextMenus.create({
    id: 'removeWatermarkAll',
    title: 'Remove Star Mark from All Images',
    contexts: ['page', 'selection']
  });

  console.log('[Background] Extension installed, context menus created');
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'removeWatermark') {
    console.log('[Background] Context menu clicked (single):', info.srcUrl);
    handleAction(tab.id, 'processImage', { imageUrl: info.srcUrl });
  } else if (info.menuItemId === 'removeWatermarkAll') {
    console.log('[Background] Context menu clicked (batch)');
    handleAction(tab.id, 'processAllImages', {});
  }
});

async function handleAction(tabId, action, data) {
  try {
    await ensureContentScript(tabId);
    chrome.tabs.sendMessage(tabId, { action, ...data }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('[Background] Error:', chrome.runtime.lastError.message);
      }
    });
  } catch (error) {
    console.error('[Background] Error processing action:', error);
  }
}

// Ensure content script is loaded in the tab
async function ensureContentScript(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, { action: 'ping' }, (response) => {
      if (chrome.runtime.lastError) {
        // Content script not loaded, inject it
        console.log('[Background] Injecting content script...');
        // Inject ONNX runtime first, then content script so `ort` is defined
        chrome.scripting.executeScript({
          target: { tabId },
          files: ['lib/ort.min.js', 'content.js']
        }).then(() => {
          return chrome.scripting.insertCSS({
            target: { tabId },
            files: ['content.css']
          });
        }).then(() => {
          // Wait a bit for script to initialize
          setTimeout(resolve, 100);
        }).catch(reject);
      } else {
        resolve();
      }
    });
  });
}

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[Background] Received message:', message.action);

  if (message.action === 'getStatus') {
    // Model loads on-demand in content script
    sendResponse({
      status: 'ready',
      message: 'Ready - right-click an image to process'
    });
    return false;
  }

  if (message.action === 'loadModel') {
    // Model will be loaded in content script when first image is processed
    sendResponse({ status: 'ready' });

    // Broadcast progress for UI feedback
    broadcastProgress(100, 'Ready to process images');
    broadcastStatus('ready', 'Model loads on first use');
    return false;
  }

  if (message.action === 'fetchImage') {
    // Fetch image with service worker privileges (CORS workaround)
    fetchImageAsBase64(message.url)
      .then(data => sendResponse({ success: true, ...data }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep channel open for async response
  }

  // Process image in offscreen document (for content script right-click)
  if (message.action === 'processInOffscreen') {
    processInOffscreen(message.imageData, message.width, message.height)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep channel open for async response
  }

  return false;
});

// Offscreen document management
let offscreenCreating = null;

async function ensureOffscreenDocument() {
  // Check if offscreen document already exists
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });

  if (existingContexts.length > 0) {
    return;
  }

  // Create offscreen document
  if (offscreenCreating) {
    await offscreenCreating;
    return;
  }

  offscreenCreating = chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['DOM_PARSER'],
    justification: 'Process images for star mark removal using MI-GAN'
  });

  await offscreenCreating;
  offscreenCreating = null;
  console.log('[Background] Offscreen document created');
}

async function processInOffscreen(imageData, width, height) {
  await ensureOffscreenDocument();

  // Send processing request to offscreen document
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({
      type: 'processImage',
      imageData: imageData,
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

// Fetch image as base64 (bypasses CORS for content script)
async function fetchImageAsBase64(url) {
  console.log('[Background] Fetching image:', url);

  let response;

  // Try different fetch modes
  const fetchModes = [
    { mode: 'cors', credentials: 'omit' },
    { mode: 'no-cors', credentials: 'omit' },
    { credentials: 'include' },
    {}
  ];

  for (const options of fetchModes) {
    try {
      response = await fetch(url, options);
      if (response.ok || response.type === 'opaque') {
        console.log('[Background] Fetch succeeded with options:', options);
        break;
      }
    } catch (e) {
      console.log('[Background] Fetch failed with options:', options, e.message);
    }
  }

  if (!response) {
    throw new Error('All fetch attempts failed');
  }

  // For opaque responses (no-cors), we can't read the body directly
  // But service workers should have more privileges
  if (!response.ok && response.type !== 'opaque') {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const blob = await response.blob();

  if (blob.size === 0) {
    throw new Error('Empty response');
  }

  console.log('[Background] Got blob, size:', blob.size, 'type:', blob.type);

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      resolve({
        data: base64,
        mimeType: blob.type || 'image/png'
      });
    };
    reader.onerror = () => reject(new Error('Failed to read image data'));
    reader.readAsDataURL(blob);
  });
}

// Broadcast status to popup
function broadcastStatus(status, message) {
  chrome.runtime.sendMessage({
    type: 'status',
    status,
    message
  }).catch(() => { }); // Ignore if popup is closed
}

// Broadcast progress to popup
function broadcastProgress(percent, text) {
  chrome.runtime.sendMessage({
    type: 'progress',
    percent,
    text
  }).catch(() => { }); // Ignore if popup is closed
}

console.log('[Background] Service worker initialized');

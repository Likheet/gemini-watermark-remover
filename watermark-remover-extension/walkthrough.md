# Watermark Remover Extension - Walkthrough

## Overview
This extension has been redesigned to feature a robust **3-Page Dashboard** powered by a persistent background **Web Worker**. The design separates concerns into Single Image, Batch Queue, and Manual Masking modes.

## 1. Extension Architecture
- **popup.html / popup.js**: Minimal entry point. Opens the dashboard in a new tab.
- **dashboard.html**: The main UI container.
  - **Single Mode**: Drag-and-drop for one image. Shows side-by-side comparison.
  - **Batch Mode**: Queue system for processing multiple images sequentially. Supports "Download All" and "Download ZIP".
  - **Manual Mode**: Canvas-based painting tool to create custom masks for the inpainting model.
- **dashboard.js**: The central controller.
  - Manages UI state (`currentPage`, `isProcessing`).
  - Handles drag-and-drop events for all zones.
  - initializing and communicating with the Web Worker.
- **processor.worker.js**: The AI brain.
  - Runs in a background thread to prevent UI freezing.
  - Loads the ONNX model (`migan.onnx`).
  - Performs image preprocessing, watermark detection, and interference.
  - Sends "Theatrical" status updates to keep the user engaged during processing.

## 2. Key Features
### A. Theatrical Status Updates
The worker simulates a complex analysis process with staggered messages:
1. "Analyzing image structure..." (10%)
2. "Identifying watermark patterns..." (30%)
3. "Generating inpaint mask..." (50%)
4. "Synthesizing clean texture..." (75%)
5. "Finalizing..." (90%)

This provides immediate visual feedback via a progress bar, even before the heavy AI computation finishes.

### B. Batch Processing Queue
- Users can drop multiple files at once.
- Files enter a `queue` array with `pending` status.
- **Sequential Processing**: The system processes one image at a time to manage memory and ensure stability.
- **Auto-Continue**: When one image finishes, the `handleResult` function automatically triggers the next item in the queue.
- **Download Options**: Individual downloads or a generated ZIP file of all results.

### C. Manual Masking
- Users can paint over the watermark if the auto-detection misses it.
- A red semi-transparent brush shows where the user is painting.
- The mask is sent as a separate buffer to the worker, bypassing the detection step.

## 3. State Management
- **Tab Locking**: While `isProcessing` is true, navigation tabs are visually disabled (opacity 0.5) and unclickable to prevent state corruption.
- **Error Handling**: Worker errors are caught and displayed. In batch mode, a failed item is marked "Error" and the queue continues to the next item.

## 4. Usage
1. Click the extension icon to open the Dashboard.
2. **Single**: Drop an image, click "Remove Watermark".
3. **Batch**: Drop multiple images, click "Process Batch". Wait for all to finish, then download.
4. **Manual**: Drop an image, paint over the watermark, click "Remove Object".

## 5. Technical Notes
- **ONNX Runtime**: Uses `ort.min.js` and `ort-wasm-simd.wasm`.
- **Memory**: Images are processed using `OffscreenCanvas` and `ImageBitmap` to keep the main thread responsive.

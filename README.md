# Star Mark Remover

Star Mark Remover is a browser extension and web tool designed to automatically detect and remove specific "star" shaped watermarks from AI-generated images. It runs entirely in your browser using local AI inference, ensuring your images never leave your device for processing.

## Project Overview

This tool was built to address the need for a privacy-focused, client-side solution for cleaning up images. Unlike many other tools that rely on cloud APIs, Star Mark Remover leverages the power of your own hardware to perform complex image inpainting tasks. This approach offers several benefits:
- **Privacy**: Your photos are processed locally and are never uploaded to a remote server.
- **Speed**: No latency from network uploads or downloads; processing happens instantly.
- **Cost**: It's free to use and doesn't incur server costs for the developer.

## Key Features

- **Single Image Mode**: Drag and drop an individual image to instantly remove watermarks.
- **Batch Processing**: Queue up multiple images to be processed automatically in sequence.
- **Manual Masking**: A dedicated editor that allows you to manually paint over unwanted objects or artifacts if the automatic detection misses something.
- **Privacy-First**: Zero data collection. Everything mimics a native desktop application's behavior within your browser.

## Technical Architecture and Decisions

The core of this project relies on **ONNX Runtime Web**, a cross-platform inference engine for machine learning models.

### Why ONNX Runtime?
We chose ONNX Runtime because it allows us to run pre-trained PyTorch models (exported to ONNX format) directly in the browser via WebAssembly (WASM) and WebGL. This enables near-native performance for deep learning tasks without requiring the user to install Python or complex dependencies.

### Inpainting Model
The underlying AI is based on a GAN-based inpainting architecture (specifically adapted from MI-GAN). It has been trained to recognize the specific visual patterns of the star watermarks and generate contextually accurate fill for the occluded areas.

### Chrome Extension (Manifest V3)
The project is structured as a Chrome Extension using Manifest V3 standards. This was chosen to:
- Provide seamless integration into the user's browsing workflow.
- Allow for offline functionality.
- Utilize the `offscreen` or `worker` capabilities for handling heavy computational tasks without freezing the UI thread.

### Tech Stack
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript. We avoided heavy frameworks (like React or Vue) for the extension to keep the bundle size small and startup times instant.
- **Inference**: `ort.min.js` (ONNX Runtime Web).
- **Image Processing**: Canvas API for image manipulation, resizing, and pixel data management.

## Installation

1. Download the repository or the provided ZIP file (latest release).
2. Open Chrome and navigate to `chrome://extensions/`.
3. Enable "Developer mode" in the top right corner.
4. Click "Load unpacked" and select the `watermark-remover-extension` folder.
5. The extension is now installed and ready to use.

## License

This project is open-source. Feel free to explore the code, submit issues, or contribute improvements.

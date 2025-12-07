# Star Mark Remover - Browser Extension

A Chrome/Edge extension that removes AI star marks from images using MI-GAN deep learning inpainting.

## Features

- 🎯 **Automatic Detection**: Uses template matching to find the 4-pointed star mark in the bottom-right corner
- 🧠 **AI-Powered**: Uses MI-GAN (~30MB) for high-quality inpainting
- 🖱️ **Right-Click Menu**: Right-click any image and select "Remove Star Mark"
- 📄 **Batch Processing**: Process all images on a page with one click
- 🔒 **Privacy-Focused**: All processing happens locally in your browser using ONNX Runtime Web
- ⚡ **On-Demand Loading**: Model loads automatically when you first process an image

## Installation

### From Source (Developer Mode)

1. Download or clone this repository
2. Open Chrome/Edge and go to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right)
4. Click "Load unpacked"
5. Select the `watermark-remover-extension` folder
6. The extension icon should appear in your toolbar

## Usage

### Remove Star Mark from Single Image:
1. Right-click on any image with a star mark
2. Select "Remove Star Mark"
3. Wait for the model to load (first time only, ~30MB download)
4. The image will be processed and replaced in-place

### Process All Images on Page:
1. Click the extension icon in the toolbar
2. Click "Process Images on Page"
3. All images with detected star marks will be processed

## How It Works

1. **Detection**: Uses template matching (ported from Python/OpenCV) to locate the star mark
2. **Masking**: Creates a precise star-shaped mask around the detected star mark
3. **Inpainting**: Uses MI-GAN ONNX model via ONNX Runtime Web for in-browser inference
4. **Blending**: Smoothly blends the inpainted region with the original image

## Technical Details

- **Model**: MI-GAN (Mobile-friendly Image Inpainting GAN)
- **Size**: ~30MB (vs ~200MB for LaMa)
- **Runtime**: ONNX Runtime Web (WebAssembly)
- **Processing**: Entirely client-side, no server required

## Privacy

This extension:
- Does NOT send any images to external servers
- Does NOT collect any user data
- All AI processing happens locally in your browser
- The only network request is downloading the AI model from HuggingFace

## Limitations

- Works best on images where the star mark is clearly visible
- The star mark should be in the bottom-right area
- Very small images (< 200px) are skipped
- CORS restrictions may prevent processing some images

## Credits

- MI-GAN model: [Picsart AI Research](https://github.com/Picsart-AI-Research/MI-GAN)
- ONNX Runtime Web: [Microsoft](https://github.com/microsoft/onnxruntime)
- Inspired by [inpaint-web](https://github.com/lxfater/inpaint-web)

## License

MIT License

"""
Test MI-GAN inpainting model - Version 2.
Using the raw migan.onnx model with proper preprocessing like IOPaint does.
"""

import os
import numpy as np
from PIL import Image
import cv2
import gradio as gr

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")

from huggingface_hub import hf_hub_download


def norm_img(img):
    """Normalize image to [0, 1] float32 and convert to CHW format."""
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


def pad_img_to_modulo(img, mod=512, square=True):
    """Pad image to be divisible by mod, optionally to square."""
    if img.ndim == 2:
        h, w = img.shape
        channels = 1
    else:
        h, w, channels = img.shape
    
    if square:
        max_dim = max(h, w)
        target_h = ((max_dim + mod - 1) // mod) * mod
        target_w = target_h
    else:
        target_h = ((h + mod - 1) // mod) * mod
        target_w = ((w + mod - 1) // mod) * mod
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    if img.ndim == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    return padded, h, w


class MiganInpainter:
    """MI-GAN inpainting using raw migan.onnx with IOPaint-style preprocessing.
    
    This uses the raw model, not the pipeline, for better control.
    """
    
    def __init__(self):
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using GPU (CUDA) for inference")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU for inference (GPU not available)")
        
        # Download raw MI-GAN ONNX model (not the pipeline)
        print("Downloading MI-GAN ONNX model... (~30MB)")
        model_path = hf_hub_download(
            repo_id="lxfater/inpaint-web",
            filename="migan.onnx"
        )
        print(f"Model downloaded to: {model_path}")
        
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        print(f"Model inputs: {[(inp.name, inp.shape, inp.type) for inp in self.inputs]}")
        print(f"Model outputs: {[(out.name, out.shape, out.type) for out in self.outputs]}")
        print("MI-GAN model loaded successfully!")
    
    def _get_crop_region(self, mask, margin=128):
        """Get bounding box of mask with margin."""
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 127)
        
        if len(xs) == 0 or len(ys) == 0:
            return None
        
        x1 = max(0, xs.min() - margin)
        y1 = max(0, ys.min() - margin)
        x2 = min(w, xs.max() + margin)
        y2 = min(h, ys.max() + margin)
        
        return x1, y1, x2, y2
    
    def _inpaint_region(self, image, mask):
        """Inpaint a region using MI-GAN.
        
        Args:
            image: RGB numpy array [H, W, 3]
            mask: Grayscale numpy array [H, W], 255 = hole
        
        Returns:
            Inpainted RGB numpy array
        """
        # Pad to 512x512 square (MI-GAN requirement)
        padded_image, orig_h, orig_w = pad_img_to_modulo(image, mod=512, square=True)
        padded_mask, _, _ = pad_img_to_modulo(mask, mod=512, square=True)
        
        # Normalize image: [0,1] -> [-1, 1]
        img_norm = norm_img(padded_image)  # [3, H, W] in [0, 1]
        img_norm = img_norm * 2 - 1  # [-1, 1]
        
        # Normalize mask: binary, then [0, 1]
        mask_binary = (padded_mask > 127).astype(np.float32)  # 1 = hole
        
        # Create erased image (mask out the hole region)
        erased_img = img_norm * (1 - mask_binary)  # Zero out hole pixels
        
        # MI-GAN input: 4 channels = [0.5 - mask, erased_R, erased_G, erased_B]
        # The first channel is 0.5 where known, -0.5 where hole
        mask_channel = 0.5 - mask_binary  # 0.5=keep, -0.5=hole
        
        input_tensor = np.concatenate([
            mask_channel[np.newaxis, ...],  # [1, H, W]
            erased_img                       # [3, H, W]
        ], axis=0)  # [4, H, W]
        
        input_tensor = input_tensor[np.newaxis, ...].astype(np.float32)  # [1, 4, H, W]
        
        # Run inference
        input_name = self.inputs[0].name
        output_name = self.outputs[0].name
        
        outputs = self.session.run([output_name], {input_name: input_tensor})
        output = outputs[0]  # [1, 3, H, W] in [-1, 1]
        
        # Convert output from [-1, 1] to [0, 255]
        output = output[0]  # [3, H, W]
        output = (output + 1) / 2  # [-1, 1] -> [0, 1]
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # [H, W, 3]
        
        # Crop back to original size
        output = output[:orig_h, :orig_w, :]
        
        return output
    
    def __call__(self, image, mask):
        """Run inpainting with crop-based approach for better quality.
        
        Args:
            image: PIL Image or numpy array (RGB)
            mask: PIL Image or numpy array (grayscale, 255 = hole)
        
        Returns:
            PIL Image with inpainted result
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        h, w = image.shape[:2]
        
        # Get crop region around the mask
        crop_box = self._get_crop_region(mask, margin=64)
        
        if crop_box is None:
            print("No mask detected, returning original")
            return Image.fromarray(image)
        
        x1, y1, x2, y2 = crop_box
        print(f"Crop region: ({x1},{y1}) to ({x2},{y2}), size: {x2-x1}x{y2-y1}")
        
        # Crop the region
        crop_image = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()
        
        # Resize to max 512 for MI-GAN
        crop_h, crop_w = crop_image.shape[:2]
        max_dim = max(crop_h, crop_w)
        
        if max_dim > 512:
            scale = 512 / max_dim
            new_h, new_w = int(crop_h * scale), int(crop_w * scale)
            crop_image_resized = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            crop_mask_resized = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            crop_image_resized = crop_image
            crop_mask_resized = crop_mask
            new_h, new_w = crop_h, crop_w
        
        print(f"Processing at size: {new_w}x{new_h}")
        
        # Run MI-GAN inpainting
        inpainted = self._inpaint_region(crop_image_resized, crop_mask_resized)
        
        # Resize back if needed
        if max_dim > 512:
            inpainted = cv2.resize(inpainted, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        # Blend: only paste the inpainted pixels where mask is active
        result = image.copy()
        crop_result = result[y1:y2, x1:x2]
        
        # Use mask to blend (smooth blending at edges)
        mask_float = crop_mask.astype(np.float32) / 255.0
        
        # Smooth the mask edges for better blending
        mask_blurred = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_blurred = mask_blurred[:, :, np.newaxis]  # [H, W, 1] - add channel dim AFTER blur
        
        blended = (inpainted * mask_blurred + crop_result * (1 - mask_blurred)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
        
        return Image.fromarray(result)


def create_star_template(size=48):
    """Create a 4-pointed star template for template matching."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    outer_radius = size // 2 - 2
    inner_radius = size // 6
    
    points = []
    for i in range(8):
        angle = i * np.pi / 4 - np.pi / 2
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        x = center + int(r * np.cos(angle))
        y = center + int(r * np.sin(angle))
        points.append([x, y])
    
    points = np.array(points, np.int32)
    cv2.fillPoly(img, [points], 255)
    return img


def detect_star_watermark(image):
    """Detect the white 4-pointed star watermark using template matching."""
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Focus on bottom-right area
    search_h = min(300, h // 2)
    search_w = min(300, w // 2)
    
    roi_y = h - search_h
    roi_x = w - search_w
    roi = gray[roi_y:h, roi_x:w]
    
    print(f"Searching in bottom-right region: ({roi_x},{roi_y}) to ({w},{h})")
    
    base_template = create_star_template(48)
    scales = np.linspace(0.3, 2.0, 20)
    
    best_val = 0
    best_loc = None
    best_size = None
    
    for scale in scales:
        new_size = int(48 * scale)
        if new_size < 10 or new_size > min(roi.shape[0], roi.shape[1]):
            continue
        
        template = cv2.resize(base_template, (new_size, new_size))
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_size = new_size
    
    print(f"Template matching best score: {best_val:.3f}")
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if best_val > 0.35 and best_loc is not None:
        star_x = roi_x + best_loc[0]
        star_y = roi_y + best_loc[1]
        
        # Tight mask with minimal padding
        pad = max(3, int(best_size * 0.1))
        
        # Create star-shaped mask
        star_mask = create_star_template(best_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
        star_mask = cv2.dilate(star_mask, kernel, iterations=1)
        
        # Place at detected position
        y1 = star_y
        y2 = min(h, star_y + best_size)
        x1 = star_x
        x2 = min(w, star_x + best_size)
        
        mask_y2 = y2 - y1
        mask_x2 = x2 - x1
        
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = star_mask[:mask_y2, :mask_x2]
        
        print(f"DETECTED star at ({star_x},{star_y}) size {best_size}x{best_size}")
    else:
        print(f"Template matching confidence too low ({best_val:.3f}), using fallback...")
        
        # Fallback position
        star_size = max(20, min(50, int(min(w, h) * 0.02)))
        center_x = w - 20 - star_size // 2
        center_y = h - 100
        
        star_mask = create_star_template(star_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        star_mask = cv2.dilate(star_mask, kernel, iterations=1)
        
        y1 = max(0, center_y - star_size // 2)
        y2 = min(h, center_y + star_size // 2)
        x1 = max(0, center_x - star_size // 2)
        x2 = min(w, center_x + star_size // 2)
        
        mask_h = y2 - y1
        mask_w = x2 - x1
        if mask_h > 0 and mask_w > 0:
            resized_mask = cv2.resize(star_mask, (mask_w, mask_h))
            mask[y1:y2, x1:x2] = resized_mask
        
        print(f"Using fallback position: center=({center_x},{center_y})")
    
    detected_pixels = np.count_nonzero(mask)
    print(f"Mask covers {detected_pixels} pixels")
    
    return mask


# Initialize model
print("Loading MI-GAN model...")
migan = MiganInpainter()
print("MI-GAN model loaded!")


def remove_watermark(image):
    """Remove watermark using MI-GAN."""
    if image is None:
        return None
    
    image = np.array(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    mask = detect_star_watermark(image)
    
    print("Running MI-GAN inpainting...")
    result = migan(image, mask)
    print("MI-GAN inpainting complete!")
    
    return np.array(result)


def compare_with_mask(image):
    """Show detected mask and result."""
    if image is None:
        return None, None
    
    image = np.array(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    mask = detect_star_watermark(image)
    
    # Visualize mask
    mask_viz = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask
    mask_viz = cv2.addWeighted(mask_viz, 0.7, mask_colored, 0.3, 0)
    
    # Run inpainting
    result = remove_watermark(image)
    
    return mask_viz, result


def main():
    with gr.Blocks(title="MI-GAN Inpainting Test v2") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>MI-GAN Inpainting Test v2</h1>
            <p>Using raw MI-GAN with IOPaint-style preprocessing</p>
            <p style="color: #666;">Crop-based inpainting for better quality</p>
        </div>
        """)
        
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Image with Star Watermark")
        
        remove_btn = gr.Button("Remove Watermark", variant="primary")
        
        with gr.Row():
            with gr.Column():
                mask_output = gr.Image(label="Detected Region (Red)")
            with gr.Column():
                result_output = gr.Image(label="MI-GAN Result")
        
        remove_btn.click(
            fn=compare_with_mask,
            inputs=[input_image],
            outputs=[mask_output, result_output]
        )
    
    demo.launch(server_port=7864)


if __name__ == "__main__":
    print("MI-GAN Inpainting Test v2")
    print("=========================")
    print("Using raw MI-GAN with proper preprocessing")
    print("")
    main()

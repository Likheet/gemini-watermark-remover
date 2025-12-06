"""
Benchmark script to identify watermark removal bottlenecks.
"""

import time
import numpy as np
from PIL import Image
import cv2

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


def detect_star_watermark_original(image):
    """ORIGINAL: Detect the white 4-pointed star watermark using template matching."""
    h, w = image.shape[:2]
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    search_h = min(300, h // 2)
    search_w = min(300, w // 2)
    
    roi_y = h - search_h
    roi_x = w - search_w
    roi = gray[roi_y:h, roi_x:w]
    
    base_template = create_star_template(48)
    scales = np.linspace(0.3, 2.0, 20)  # 20 scales!
    
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
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if best_val > 0.35 and best_loc is not None:
        star_x = roi_x + best_loc[0]
        star_y = roi_y + best_loc[1]
        
        pad = max(3, int(best_size * 0.1))
        
        star_mask = create_star_template(best_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
        star_mask = cv2.dilate(star_mask, kernel, iterations=1)
        
        y1 = star_y
        y2 = min(h, star_y + best_size)
        x1 = star_x
        x2 = min(w, star_x + best_size)
        
        mask_y2 = y2 - y1
        mask_x2 = x2 - x1
        
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = star_mask[:mask_y2, :mask_x2]
    
    return mask, best_val


def detect_star_watermark_optimized(image):
    """OPTIMIZED: Fewer scales + smaller search region."""
    h, w = image.shape[:2]
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Smaller search region
    search_h = min(150, h // 3)
    search_w = min(150, w // 3)
    
    roi_y = h - search_h
    roi_x = w - search_w
    roi = gray[roi_y:h, roi_x:w]
    
    base_template = create_star_template(48)
    
    # OPTIMIZATION 1: Only 8 scales instead of 20
    scales = np.linspace(0.5, 1.5, 8)
    
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
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if best_val > 0.35 and best_loc is not None:
        star_x = roi_x + best_loc[0]
        star_y = roi_y + best_loc[1]
        
        pad = max(3, int(best_size * 0.1))
        
        star_mask = create_star_template(best_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
        star_mask = cv2.dilate(star_mask, kernel, iterations=1)
        
        y1 = star_y
        y2 = min(h, star_y + best_size)
        x1 = star_x
        x2 = min(w, star_x + best_size)
        
        mask_y2 = y2 - y1
        mask_x2 = x2 - x1
        
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = star_mask[:mask_y2, :mask_x2]
    
    return mask, best_val


class MiganInpainter:
    def __init__(self):
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using GPU (CUDA) for inference")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU for inference (GPU not available)")
        
        print("Downloading MI-GAN ONNX model...")
        model_path = hf_hub_download(
            repo_id="lxfater/inpaint-web",
            filename="migan.onnx"
        )
        
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        print("MI-GAN model loaded!")
    
    def _inpaint_region(self, image, mask):
        """Inpaint a region using MI-GAN."""
        padded_image, orig_h, orig_w = pad_img_to_modulo(image, mod=512, square=True)
        padded_mask, _, _ = pad_img_to_modulo(mask, mod=512, square=True)
        
        img_norm = norm_img(padded_image)
        img_norm = img_norm * 2 - 1
        
        mask_binary = (padded_mask > 127).astype(np.float32)
        erased_img = img_norm * (1 - mask_binary)
        mask_channel = 0.5 - mask_binary
        
        input_tensor = np.concatenate([
            mask_channel[np.newaxis, ...],
            erased_img
        ], axis=0)
        
        input_tensor = input_tensor[np.newaxis, ...].astype(np.float32)
        
        input_name = self.inputs[0].name
        output_name = self.outputs[0].name
        
        outputs = self.session.run([output_name], {input_name: input_tensor})
        output = outputs[0]
        
        output = output[0]
        output = (output + 1) / 2
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        
        output = output[:orig_h, :orig_w, :]
        
        return output
    
    def __call__(self, image, mask):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        h, w = image.shape[:2]
        
        # Get crop region
        ys, xs = np.where(mask > 127)
        if len(xs) == 0 or len(ys) == 0:
            return image
        
        margin = 64
        x1 = max(0, xs.min() - margin)
        y1 = max(0, ys.min() - margin)
        x2 = min(w, xs.max() + margin)
        y2 = min(h, ys.max() + margin)
        
        crop_image = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()
        
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
        
        inpainted = self._inpaint_region(crop_image_resized, crop_mask_resized)
        
        if max_dim > 512:
            inpainted = cv2.resize(inpainted, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        result = image.copy()
        crop_result = result[y1:y2, x1:x2]
        
        mask_float = crop_mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_blurred = mask_blurred[:, :, np.newaxis]
        
        blended = (inpainted * mask_blurred + crop_result * (1 - mask_blurred)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
        
        return result


def run_benchmark():
    print("=" * 60)
    print("WATERMARK REMOVAL BENCHMARK")
    print("=" * 60)
    
    # Create a test image
    print("\nCreating test image (1024x1024)...")
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Add a white star in bottom-right
    star = create_star_template(40)
    test_image[1024-120:1024-80, 1024-50:1024-10] = 255
    
    # Load model
    print("\n1. LOADING MODEL...")
    t0 = time.time()
    migan = MiganInpainter()
    model_load_time = time.time() - t0
    print(f"   Model load time: {model_load_time:.2f}s")
    
    # Benchmark detection - ORIGINAL
    print("\n2. STAR DETECTION (ORIGINAL - 20 scales)...")
    t0 = time.time()
    mask_orig, score_orig = detect_star_watermark_original(test_image)
    detect_original_time = time.time() - t0
    print(f"   Detection time: {detect_original_time:.3f}s, score: {score_orig:.3f}")
    
    # Benchmark detection - OPTIMIZED
    print("\n3. STAR DETECTION (OPTIMIZED - 8 scales)...")
    t0 = time.time()
    mask_opt, score_opt = detect_star_watermark_optimized(test_image)
    detect_optimized_time = time.time() - t0
    print(f"   Detection time: {detect_optimized_time:.3f}s, score: {score_opt:.3f}")
    print(f"   Speedup: {detect_original_time/detect_optimized_time:.1f}x faster")
    
    # Benchmark preprocessing
    print("\n4. PREPROCESSING...")
    t0 = time.time()
    crop_image = test_image[900:1024, 900:1024].copy()
    crop_mask = mask_orig[900:1024, 900:1024].copy()
    padded_image, _, _ = pad_img_to_modulo(crop_image, mod=512, square=True)
    padded_mask, _, _ = pad_img_to_modulo(crop_mask, mod=512, square=True)
    preprocess_time = time.time() - t0
    print(f"   Preprocessing time: {preprocess_time:.3f}s")
    print(f"   Padded size: {padded_image.shape}")
    
    # Benchmark MI-GAN inference
    print("\n5. MI-GAN INFERENCE...")
    # Warm-up run
    _ = migan(test_image, mask_orig)
    
    # Timed runs
    times = []
    for i in range(3):
        t0 = time.time()
        result = migan(test_image, mask_orig)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f}s")
    
    avg_inference = sum(times) / len(times)
    print(f"   Average inference time: {avg_inference:.3f}s")
    
    # Total time estimate
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Detection (original):  {detect_original_time:.3f}s")
    print(f"Detection (optimized): {detect_optimized_time:.3f}s")
    print(f"MI-GAN inference:      {avg_inference:.3f}s")
    print(f"")
    print(f"Total (original):   {detect_original_time + avg_inference:.3f}s")
    print(f"Total (optimized):  {detect_optimized_time + avg_inference:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

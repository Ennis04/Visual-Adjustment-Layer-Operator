import cv2
import numpy as np

def apply_brightness(img, value):
    value = float(np.clip(value, -100, 100))
    img_f = img.astype(np.float32) / 255.0

    if value >= 0:
        # brighten: scale up
        factor = 1.0 + (value / 100.0)  # 1..2
        out = img_f * factor
    else:
        # darken: gamma > 1 darkens smoothly
        gamma = 1.0 + (-value / 100.0)  # 1..2
        out = img_f ** gamma

    out = np.clip(out, 0, 1) * 255.0
    return out.astype(np.uint8)


def apply_sharpness(img, value):
    value = np.clip(value, -100, 100)

    if value == 0:
        return img

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)

    if value > 0:
        # Sharpen
        alpha = value / 100.0
        return cv2.addWeighted(img, 1 + alpha, blur, -alpha, 0)
    else:
        # Soften
        alpha = abs(value) / 100.0
        return cv2.addWeighted(img, 1 - alpha, blur, alpha, 0)


def apply_blur(img, value):
    if value <= 0:
        return img
    k = max(1, (value // 10) * 2 + 1)
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_noise_reduction(img, value):
    if value <= 0:
        return img
    k = max(1, (value // 10) * 2 + 1)
    return cv2.medianBlur(img, k)

def apply_rgb(img, r, g, b):
    img = img.astype(np.float32)

    # OpenCV uses BGR order
    img[:, :, 2] *= (1 + r / 100.0)  # Red
    img[:, :, 1] *= (1 + g / 100.0)  # Green
    img[:, :, 0] *= (1 + b / 100.0)  # Blue

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_preset(img, preset: str):
    if preset == "dramatic-warm":
        # warm = boost red, reduce blue slightly, add contrast
        img = cv2.convertScaleAbs(img, alpha=1.15, beta=5)
        b, g, r = cv2.split(img)
        r = cv2.convertScaleAbs(r, alpha=1.10, beta=0)
        b = cv2.convertScaleAbs(b, alpha=0.92, beta=0)
        return cv2.merge([b, g, r])

    if preset == "dramatic-cool":
        img = cv2.convertScaleAbs(img, alpha=1.15, beta=5)
        b, g, r = cv2.split(img)
        b = cv2.convertScaleAbs(b, alpha=1.10, beta=0)
        r = cv2.convertScaleAbs(r, alpha=0.92, beta=0)
        return cv2.merge([b, g, r])

    if preset == "noir":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=-10)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img

def remove_background(img, rect_pad=10, iter_count=5):
    """
    Remove background using GrabCut, then refine mask for smoother edges.
    Returns BGRA image with alpha channel.

    rect_pad: padding from image border for initial rectangle
    iter_count: GrabCut iterations
    """
    h, w = img.shape[:2]

    # Initial mask (all set to "probably background")
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Rectangle slightly inside image borders
    x = rect_pad
    y = rect_pad
    rw = max(1, w - 2 * rect_pad)
    rh = max(1, h - 2 * rect_pad)
    rect = (x, y, rw, rh)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # Binary mask: foreground/prob foreground = 1, else 0
    mask_bin = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1, 0
    ).astype(np.uint8)

    # --- Refinement stage ---
    # 1) Remove noise (open) and fill small holes (close)
    k = max(3, int(min(h, w) * 0.01) | 1)  # odd kernel size scaled to image
    kernel = np.ones((k, k), np.uint8)

    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) Slight erosion to reduce background halos (optional but useful)
    erode_k = max(3, (k // 2) | 1)
    erode_kernel = np.ones((erode_k, erode_k), np.uint8)
    mask_clean = cv2.erode(mask_clean, erode_kernel, iterations=1)

    # 3) Soft alpha edges
    alpha = (mask_clean * 255).astype(np.uint8)

    blur_k = max(3, int(min(h, w) * 0.01) | 1)  # odd
    alpha = cv2.GaussianBlur(alpha, (blur_k, blur_k), 0)

    # Apply alpha to original image
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, r, alpha])


def apply_crop(img, crop):
    """
    crop dict:
      enabled: bool
      x,y,w,h in normalized [0..1]
    """
    if not crop or not crop.get("enabled", False):
        return img

    H, W = img.shape[:2]
    x = int(np.clip(crop.get("x", 0), 0, 1) * W)
    y = int(np.clip(crop.get("y", 0), 0, 1) * H)
    cw = int(np.clip(crop.get("w", 1), 0, 1) * W)
    ch = int(np.clip(crop.get("h", 1), 0, 1) * H)

    cw = max(1, min(cw, W - x))
    ch = max(1, min(ch, H - y))

    return img[y:y+ch, x:x+cw]

def process_image(img, params):
    img = apply_brightness(img, params.get("brightness", 0))
    img = apply_sharpness(img, params.get("sharpness", 0))
    img = apply_noise_reduction(img, params.get("denoise", 0))
    img = apply_blur(img, params.get("blur", 0))

    preset = params.get("preset", "none")
    if preset != "none":
        if preset == "mono":
            img = apply_grayscale(img)
        else:
            img = apply_preset(img, preset)

    # IMPORTANT RULE YOU WANTED:
    # "none" does NOT reset RGB sliders. RGB sliders are ALWAYS applied unless preset is noir/mono
    if preset in ("none", "dramatic-warm", "dramatic-cool"):
        img = apply_rgb(img, params.get("red", 0), params.get("green", 0), params.get("blue", 0))


    # Preset stage (acts like a "look" layer)
    img = apply_crop(img, params.get("crop", None))

    return img



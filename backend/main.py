import cv2
import numpy as np


def apply_brightness(img, value):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=value)


def apply_sharpness(img, value):
    if value <= 0:
        return img
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + value / 10, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def apply_noise_reduction(img, value):
    if value <= 0:
        return img
    k = max(1, (value // 10) * 2 + 1)
    return cv2.medianBlur(img, k)


def apply_rgb(img, r, g, b):
    """
    r, g, b are in range [-100, 100]
    """
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

def _apply_contrast_saturation(bgr, contrast=1.0, saturation=1.0):
    # contrast in BGR space
    out = cv2.convertScaleAbs(bgr, alpha=contrast, beta=0)

    if abs(saturation - 1.0) > 1e-6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out

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

def remove_background(img):
    """
    Remove background using GrabCut.
    Returns image with transparent background (BGRA).
    """
    h, w = img.shape[:2]

    # Initial mask
    mask = np.zeros((h, w), np.uint8)

    # Rectangle slightly inside image borders
    rect = (10, 10, w - 20, h - 20)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Convert mask to binary
    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype("uint8")

    # Apply mask
    result = img * mask2[:, :, np.newaxis]

    # Convert to BGRA
    b, g, r = cv2.split(result)
    alpha = (mask2 * 255).astype(np.uint8)

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



import cv2
import numpy as np
import os

# Configurable decision threshold for shape matching. Lower is stricter.
MATCH_THRESHOLD = 1.0

REFERENCE_IMAGES = {
    "Object 1": "images/ref_obj1.jpg",
    "Object 2": "images/ref_obj2.jpg",
    "Object 3": "images/ref_obj3.jpg",
    "Object 4": "images/ref_obj4.jpg",
}

# One or more test image paths. Update to match your provided files.
TEST_IMAGES = [
    "images/test1.jpg",
    "images/test2.jpg",
    "images/test3.jpg", 
    "images/test4.jpg",
    "images/test5.jpg"
]

# Maximum display width for test images.
MAX_DISPLAY_WIDTH = 400


def load_image_safe(path):
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: could not read image: {path}")
    return img


def resize_for_display(image, max_width=MAX_DISPLAY_WIDTH):
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = float(max_width) / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def overlay_classification_text(image, object_label, shape_type):
    text1 = f"{object_label}"
    text2 = f"{shape_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 10

    (w1, h1), _ = cv2.getTextSize(text1, font, scale, thickness)
    (w2, h2), _ = cv2.getTextSize(text2, font, scale, thickness)
    box_width = max(w1, w2) + margin * 2
    box_height = h1 + h2 + margin * 3

    cv2.rectangle(image, (0, 0), (box_width, box_height), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text1, (margin, margin + h1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(image, text2, (margin, margin + h1 + margin + h2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Illumination correction with morphological opening.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(gray, background)

    # Adaptive threshold to handle lighting variation.
    thresh = cv2.adaptiveThreshold(
        corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2,
    )

    # Morphological clean-up to remove noise and close small gaps.
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    return cleaned


def get_main_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image.")
        return None 
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    print(f"Main contour area: {area:.1f}")
    if area < 50:
        print("Main contour area too small (< 50), no object detected.")
        return None
    return main_contour


def extract_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h != 0 else 0.0

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = len(approx)

    is_circle_like = circularity > 0.7 and vertex_count >= 8
    shape_type = "circle" if is_circle_like else "non-circle/bracket-like"

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "vertices": vertex_count,
        "shape_type": shape_type,
    }


def rotate_contour(contour, angle_degrees, center=None):
    pts = contour.reshape(-1, 2).astype(np.float32)
    if center is None:
        center = pts.mean(axis=0)

    rad = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]],
        dtype=np.float32,
    )
    rotated = np.dot(pts - center, rotation_matrix.T) + center
    return rotated.reshape(-1, 1, 2).astype(np.int32)


def scale_contour(contour, scale, center=None):
    pts = contour.reshape(-1, 2).astype(np.float32)
    if center is None:
        center = pts.mean(axis=0)
    scaled = (pts - center) * scale + center
    return scaled.reshape(-1, 1, 2).astype(np.int32)


def flip_contour(contour, flip_code, center=None):
    pts = contour.reshape(-1, 2).astype(np.float32)
    if center is None:
        center = pts.mean(axis=0)

    flipped = pts.copy()
    if flip_code == 0:
        flipped[:, 1] = 2 * center[1] - pts[:, 1]
    elif flip_code == 1:
        flipped[:, 0] = 2 * center[0] - pts[:, 0]
    elif flip_code == -1:
        flipped[:, 0] = 2 * center[0] - pts[:, 0]
        flipped[:, 1] = 2 * center[1] - pts[:, 1]
    return flipped.reshape(-1, 1, 2).astype(np.int32)


def get_contour_variants(contour, rotations=(0, 90, 180, 270), scales=(0.6, 0.8, 1.0, 1.2, 1.4), allow_reflection=True):
    center = contour.reshape(-1, 2).mean(axis=0)
    variants = []
    for scale in scales:
        scaled = scale_contour(contour, scale, center)
        for angle in rotations:
            rotated = rotate_contour(scaled, angle, center)
            variants.append(rotated)
            if allow_reflection:
                variants.append(flip_contour(rotated, 0, center))
                variants.append(flip_contour(rotated, 1, center))
                variants.append(flip_contour(rotated, -1, center))
    return variants


def compare_contours(contour_a, contour_b, allow_reflection=True):
    if contour_a is None or contour_b is None:
        return float("inf")

    best_score = float("inf")
    for variant in get_contour_variants(contour_a, allow_reflection=allow_reflection):
        score = cv2.matchShapes(variant, contour_b, cv2.CONTOURS_MATCH_I1, 0.0)
        if score < best_score:
            best_score = score
    return best_score


def describe_match(test_name, test_contour, reference_contours):
    scores = []
    for name, contour in reference_contours.items():
        score = compare_contours(test_contour, contour)
        scores.append((name, score))
    scores.sort(key=lambda item: item[1])
    best_name, best_score = scores[0]
    pass_fail = "PASS" if best_score <= MATCH_THRESHOLD else "FAIL"
    return best_name, best_score, pass_fail, scores


def process_image(path, reference_contours):
    image = load_image_safe(path)
    if image is None:
        return

    processed = preprocess_image(image)
    contour = get_main_contour(processed)
    object_detected = contour is not None
    if not object_detected:
        print(f"{path}: Object detected: No")
        return
    
    print(f"{path}: Object detected: Yes")

    features = extract_shape_features(contour)
    best_name, best_score, pass_fail, scores = describe_match(path, contour, reference_contours)

    shape_label = "Circle" if features["shape_type"] == "circle" else "Non-Circle"
    display_image = image.copy()
    display_image = overlay_classification_text(display_image, best_name, shape_label)
    display_image = resize_for_display(display_image)
    cv2.imshow("Test Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\n=== Result for {path} ===")
    print(f"Area: {features['area']:.1f}")
    print(f"Perimeter: {features['perimeter']:.1f}")
    print(f"Circularity: {features['circularity']:.3f}")
    print(f"Aspect ratio: {features['aspect_ratio']:.3f}")
    print(f"Vertices: {features['vertices']}")
    print(f"Shape type: {features['shape_type']}")
    print("Match scores:")
    for name, score in scores:
        print(f"  {name}: {score:.4f}")
    print(f"Best match: {best_name}")
    print(f"Decision: {pass_fail} (threshold {MATCH_THRESHOLD})")


def build_reference_contours(reference_paths):
    contours = {}
    for name, path in reference_paths.items():
        image = load_image_safe(path)
        if image is None:
            continue
        processed = preprocess_image(image)
        contour = get_main_contour(processed)
        if contour is None:
            print(f"WARNING: no main contour found in reference {path}")
            continue
        contours[name] = contour
    return contours


def main():
    reference_contours = build_reference_contours(REFERENCE_IMAGES)
    if not reference_contours:
        print("ERROR: no valid reference contours available. Check reference image paths.")
        return

    for test_path in TEST_IMAGES:
        process_image(test_path, reference_contours)


if __name__ == "__main__":
    main()
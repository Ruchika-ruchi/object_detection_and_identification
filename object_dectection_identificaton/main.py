import cv2
import numpy as np
import os

# ================= PATHS =================
REFERENCE_IMAGES = {
    "CIRCULAR_PART": "images/ref_obj1.jpg",
    "BRACKET_PART": "images/ref_obj2.jpg"
}

# 🔥 CHANGED: single image → list
TEST_IMAGES = [
    "images/test1.jpg",
    "images/test2.jpg",
    "images/test3.jpg",
    "images/test4.jpg",
    "images/test5.jpg",
    "images/test6.jpg"
]

# ================= LOAD IMAGE =================
def load_image(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    return cv2.imread(path)

# ================= PREPROCESS =================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    corrected = cv2.divide(gray, blur, scale=255)

    thresh = cv2.adaptiveThreshold(
        corrected, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh

# ================= CONTOUR =================
def get_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 20000:
        return None

    return cv2.convexHull(cnt)

# ================= FEATURES =================
def extract_features(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 0
    if perimeter != 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    extent = area / (w * h) if (w * h) != 0 else 0

    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    vertices = len(approx)

    return circularity, aspect_ratio, extent, vertices

# ================= REFERENCE STATS =================
def build_reference_stats():
    stats = {}

    for name, path in REFERENCE_IMAGES.items():
        img = load_image(path)
        if img is None:
            continue

        binary = preprocess(img)
        cnt = get_contour(binary)

        if cnt is None:
            print(f"❌ Failed {name}")
            continue

        stats[name] = extract_features(cnt)
        print(f"✅ Loaded {name} → {stats[name]}")

    return stats

# ================= CLASSIFIER =================
def classify(features, ref_stats):
    circularity, aspect_ratio, extent, vertices = features

    circ_ref = ref_stats["CIRCULAR_PART"][0]
    bracket_ref = ref_stats["BRACKET_PART"][1]

    circ_thresh = circ_ref * 0.85
    bracket_thresh = bracket_ref * 0.85

    if extent < 0.4:
        return "UNKNOWN"

    if circularity >= circ_thresh:
        return "CIRCULAR_PART"

    if aspect_ratio >= bracket_thresh:
        return "BRACKET_PART"

    return "UNKNOWN"

# ================= PROCESS =================
def process(img_path, ref_stats):
    img = load_image(img_path)
    if img is None:
        return

    img = cv2.resize(img, (800, 600))

    binary = preprocess(img)
    cnt = get_contour(binary)

    if cnt is None:
        print(f"{img_path}: No object detected")
        return

    features = extract_features(cnt)

    result = classify(features, ref_stats)

    print(f"\n{img_path}")
    print("Features:", features)
    print("Result:", result)

    x, y, w, h = cv2.boundingRect(cnt)

    out = img.copy()
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(out, result, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= MAIN =================
def main():
    ref_stats = build_reference_stats()

    if len(ref_stats) < 2:
        print("❌ Reference loading failed")
        return

    for img_path in TEST_IMAGES:
        process(img_path, ref_stats)

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os

# ===== PATHS =====
REFERENCE_IMAGES = {
    "CIRCULAR_PART": "images/ref_obj1.jpg",
    "BRACKET_PART": "images/ref_obj2.jpg"
}

# ===== TEST IMAGES (LIST) =====
TEST_IMAGES = [
    "images/test1.jpg",
    "images/test2.jpg",
    "images/test3.jpg",
    "images/test4.jpg",
    "images/test5.jpg"
]

def load_image(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    return cv2.imread(path)

# ===== PREPROCESS =====
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

# ===== CONTOUR =====
def get_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 20000:
        return None

    return cv2.convexHull(cnt)

# ===== NORMALIZE =====
def normalize_contour(cnt):
    cnt = cnt.astype(np.float32)
    cnt -= cnt.mean(axis=0)

    norm = np.linalg.norm(cnt)
    if norm != 0:
        cnt /= norm

    return cnt.astype(np.int32)

# ===== MATCH =====
def match(cnt, ref):
    cnt = normalize_contour(cnt)
    ref = normalize_contour(ref)

    return cv2.matchShapes(cnt, ref, cv2.CONTOURS_MATCH_I1, 0)

# ===== FEATURE EXTRACTION =====
def extract_features(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 0
    if perimeter != 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    extent = area / (w * h) if (w*h) != 0 else 0

    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    vertices = len(approx)

    return circularity, aspect_ratio, extent, vertices

# ===== RULE-BASED CLASSIFIER =====
def rule_classify(features):
    circularity, aspect_ratio, extent, vertices = features

    if extent < 0.4:
        return "UNKNOWN"

    if circularity > 0.75 and vertices > 5:
        return "CIRCULAR_PART"

    if aspect_ratio > 1.4:
        return "BRACKET_PART"

    return "UNKNOWN"

# ===== MATCH CLASSIFIER =====
def match_classify(cnt, refs, thr=0.3, margin=0.01):
    scores = []

    for name, ref in refs.items():
        score = match(cnt, ref)
        scores.append((name, score))

    scores.sort(key=lambda x: x[1])

    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else None

    print("\nMatch Scores:", scores)

    if best_score > thr:
        return "UNKNOWN"

    if second_score is not None and (second_score - best_score) < margin:
        return "UNKNOWN"

    return best_name

# ===== BUILD REFERENCES =====
def build_refs():
    refs = {}

    for name, path in REFERENCE_IMAGES.items():
        img = load_image(path)
        if img is None:
            continue

        binary = preprocess(img)
        cnt = get_contour(binary)

        if cnt is not None:
            refs[name] = cnt
            print(f"✅ Loaded {name}")
        else:
            print(f"❌ Failed {name}")

    return refs

# ===== PROCESS IMAGE =====
def process_image(path, refs):
    print(f"\n===== Processing {path} =====")

    img = load_image(path)
    if img is None:
        return

    img = cv2.resize(img, (800, 600))
    binary = preprocess(img)
    cnt = get_contour(binary)

    if cnt is None:
        print("❌ No object detected")
        return

    features = extract_features(cnt)
    print("Feature Debug:", features)

    rule_result = rule_classify(features)
    print("Rule Result:", rule_result)

    match_result = match_classify(cnt, refs)
    print("Match Result:", match_result)

    if rule_result != "UNKNOWN":
        final = rule_result
    else:
        final = match_result

    print("🎯 FINAL RESULT:", final)

    output = img.copy()
    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(output, final, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Create output folder if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Save output images
    filename = os.path.basename(path).split('.')[0]
    cv2.imwrite(f"output/{filename}_result.jpg", output)
    cv2.imwrite(f"output/{filename}_binary.jpg", binary)
    print(f"✅ Saved to output/{filename}_result.jpg and output/{filename}_binary.jpg")

    cv2.imshow("Result", output)
    cv2.imshow("Binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== MAIN =====
def main():
    refs = build_refs()

    if not refs:
        print("❌ No references loaded")
        return

    for img_path in TEST_IMAGES:
        process_image(img_path, refs)

if __name__ == "__main__":
    main()
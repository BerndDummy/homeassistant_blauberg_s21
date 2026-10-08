import json
import math
import os
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import onnxruntime as ort
import requests

CORE_API = "http://supervisor/core/api"
TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
OPTIONS_PATH = Path("/data/options.json")
MODEL_PATH = "/app/mnist-12.onnx"

WARP_W = 1000
WARP_H = 600

# Normalized field windows in the rectified Aqotec display.
# Only the numeric portion is included; units are intentionally excluded.
FIELD_WINDOWS = {
    # Calibrated from the fixed Aqotec screen geometry. Earlier windows clipped
    # the lower edge of each LCD digit, which made a real 0 look like 8/1/7.
    "power": (0.69, 0.405, 0.845, 0.515),
    "flow": (0.69, 0.505, 0.845, 0.615),
}

CALIBRATION_WINDOWS = {
    "energy": (0.54, 0.285, 0.845, 0.395),
    "vorlauf": (0.66, 0.625, 0.845, 0.725),
    "ruecklauf": (0.66, 0.725, 0.845, 0.825),
    "spread": (0.70, 0.825, 0.845, 0.925),
}

TEMPLATE_ROOT = Path("/data/font_templates")

SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {TOKEN}"})


def log(event, **data):
    payload = {"event": event, **data}
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def read_options():
    defaults = {
        "camera_entity": "camera.wansview_apotec_profile1",
        "scan_interval": 30,
        "min_digit_confidence": 0.40,
        "stable_samples": 2,
        "save_debug_crops": False,
    }
    try:
        if OPTIONS_PATH.exists():
            defaults.update(json.loads(OPTIONS_PATH.read_text(encoding="utf-8")))
    except Exception as exc:
        log("options_error", error=str(exc))
    defaults["scan_interval"] = max(20, min(600, int(defaults["scan_interval"])))
    defaults["stable_samples"] = max(1, min(5, int(defaults["stable_samples"])))
    defaults["min_digit_confidence"] = max(0.0, min(1.0, float(defaults["min_digit_confidence"])))
    defaults["save_debug_crops"] = bool(defaults["save_debug_crops"])
    return defaults


def ha_get(path, timeout=20):
    response = SESSION.get(f"{CORE_API}{path}", timeout=timeout)
    response.raise_for_status()
    return response


def ha_service(domain, service, data):
    response = SESSION.post(
        f"{CORE_API}/services/{domain}/{service}",
        json=data,
        timeout=15,
    )
    response.raise_for_status()
    return response


def mqtt_publish(topic, payload, retain=False):
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    ha_service(
        "mqtt",
        "publish",
        {"topic": topic, "payload": payload, "retain": bool(retain)},
    )


def publish_discovery():
    device = {
        "identifiers": ["aqotec_digitocr_shadow"],
        "name": "Aqotec DigitOCR Shadow",
        "manufacturer": "Local",
        "model": "Digit-only ONNX shadow reader",
        "sw_version": "0.1.3",
    }
    common = {
        "state_topic": "aqotec/digitocr/state",
        "device": device,
    }
    sensors = {
        "power": {
            **common,
            "name": "Aqotec DigitOCR Leistung",
            "unique_id": "aqotec_digitocr_shadow_power",
            "value_template": "{{ value_json.power }}",
            "unit_of_measurement": "kW",
            "device_class": "power",
            "state_class": "measurement",
            "icon": "mdi:flash",
        },
        "flow": {
            **common,
            "name": "Aqotec DigitOCR Durchfluss",
            "unique_id": "aqotec_digitocr_shadow_flow",
            "value_template": "{{ value_json.flow }}",
            "unit_of_measurement": "L/h",
            "state_class": "measurement",
            "icon": "mdi:water-pump",
        },
        "power_conf": {
            **common,
            "name": "Aqotec DigitOCR Leistung Konfidenz",
            "unique_id": "aqotec_digitocr_shadow_power_confidence",
            "value_template": "{{ value_json.power_confidence }}",
            "unit_of_measurement": "%",
            "state_class": "measurement",
            "icon": "mdi:percent",
        },
        "flow_conf": {
            **common,
            "name": "Aqotec DigitOCR Durchfluss Konfidenz",
            "unique_id": "aqotec_digitocr_shadow_flow_confidence",
            "value_template": "{{ value_json.flow_confidence }}",
            "unit_of_measurement": "%",
            "state_class": "measurement",
            "icon": "mdi:percent",
        },
        "power_delta": {
            **common,
            "name": "Aqotec DigitOCR vs RapidOCR Leistung",
            "unique_id": "aqotec_digitocr_shadow_power_delta",
            "value_template": "{{ value_json.power_delta }}",
            "unit_of_measurement": "kW",
            "state_class": "measurement",
            "icon": "mdi:delta",
        },
        "flow_delta": {
            **common,
            "name": "Aqotec DigitOCR vs RapidOCR Durchfluss",
            "unique_id": "aqotec_digitocr_shadow_flow_delta",
            "value_template": "{{ value_json.flow_delta }}",
            "unit_of_measurement": "L/h",
            "state_class": "measurement",
            "icon": "mdi:delta",
        },
        "status": {
            **common,
            "name": "Aqotec DigitOCR Status",
            "unique_id": "aqotec_digitocr_shadow_status",
            "value_template": "{{ value_json.status }}",
            "icon": "mdi:eye-check-outline",
            "json_attributes_topic": "aqotec/digitocr/state",
        },
    }
    for key, config in sensors.items():
        mqtt_publish(f"homeassistant/sensor/aqotec_digitocr_shadow/{key}/config", config, retain=True)


def fetch_state(entity_id):
    try:
        data = ha_get(f"/states/{quote(entity_id, safe='._')}").json()
        return data.get("state")
    except Exception:
        return None


def fetch_float_state(entity_id):
    state = fetch_state(entity_id)
    try:
        return float(state)
    except (TypeError, ValueError):
        return None


def fetch_camera(entity_id):
    response = ha_get(f"/camera_proxy/{quote(entity_id, safe='._')}", timeout=25)
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("camera_decode_failed")
    return image


def order_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(s)],
            points[np.argmin(d)],
            points[np.argmax(s)],
            points[np.argmax(d)],
        ],
        dtype=np.float32,
    )


def fixed_screen_quad(image):
    h, w = image.shape[:2]
    return np.array(
        [
            [0.346 * w, 0.275 * h],
            [0.938 * w, 0.254 * h],
            [0.956 * w, 0.880 * h],
            [0.337 * w, 0.896 * h],
        ],
        dtype=np.float32,
    )


def detect_screen_quad(image):
    h, w = image.shape[:2]
    b, g, r = cv2.split(image)
    bi = b.astype(np.int16)
    gi = g.astype(np.int16)
    ri = r.astype(np.int16)

    # Aqotec display is distinctly blue/cyan compared with the surrounding wood.
    mask = (
        (bi > 80)
        & (bi - ri > 8)
        & (bi >= gi - 8)
    ).astype(np.uint8) * 255

    # Ignore zones where the display cannot be in this fixed camera installation.
    mask[: int(0.18 * h), :] = 0
    mask[:, : int(0.20 * w)] = 0

    close_k = max(9, int(min(h, w) * 0.025) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / float(w * h)
        if area_ratio < 0.06 or area_ratio > 0.65:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if not 1.25 <= aspect <= 2.8:
            continue

        cx, cy = rect[0]
        if cx < 0.42 * w or cy < 0.42 * h:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(rect).astype(np.float32)

        rect_area = max(1.0, rw * rh)
        rectangularity = min(1.0, area / rect_area)
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.75) / 1.75)
        score = area_ratio * (0.5 + 0.5 * rectangularity) * (0.6 + 0.4 * aspect_score)

        if score > best_score:
            best_score = score
            best = quad

    # Conservative threshold: when automatic geometry is doubtful, the fixed,
    # calibrated camera geometry is safer for a shadow sensor.
    if best is None or best_score < 0.055:
        return fixed_screen_quad(image), "fixed_fallback", 0.0

    return order_quad(best), "auto", round(min(1.0, best_score / 0.25), 3)


def warp_screen(image, quad):
    src = order_quad(quad)
    dst = np.array(
        [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (WARP_W, WARP_H))


def crop_rect(screen, rect):
    x1, y1, x2, y2 = rect
    h, w = screen.shape[:2]
    return screen[
        int(y1 * h) : int(y2 * h),
        int(x1 * w) : int(x2 * w),
    ].copy()


def field_crop(screen, name):
    return crop_rect(screen, FIELD_WINDOWS[name])


def make_binary(crop):
    # The Aqotec screen has a bright blue/cyan background and almost white
    # characters. The red channel gives substantially more digit/background
    # separation than grayscale, which overweights the bright blue background.
    red = crop[:, :, 2]
    red = cv2.GaussianBlur(red, (3, 3), 0)
    red = cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if foreground_ratio > 0.32:
        binary = cv2.bitwise_not(binary)

    # Suppress isolated camera/compression speckles but keep the LCD glyph holes.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    h, w = binary.shape[:2]
    binary[: max(1, int(0.04 * h)), :] = 0
    binary[h - max(1, int(0.04 * h)) :, :] = 0
    return binary


def digit_boxes(binary, max_digits):
    h, w = binary.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for idx in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[idx]]
        if bh < int(0.34 * h) or bh > int(0.94 * h):
            continue
        if bw < max(3, int(0.025 * w)) or bw > int(0.36 * w):
            continue
        if area < max(12, int(0.10 * bw * bh)):
            continue
        cy = y + bh / 2.0
        if cy < 0.20 * h or cy > 0.82 * h:
            continue
        boxes.append((x, y, bw, bh, area))

    # Merge components that are only tiny horizontal fragments of one glyph.
    boxes.sort(key=lambda b: b[0])
    merged = []
    for box in boxes:
        if not merged:
            merged.append(list(box))
            continue
        px, py, pw, ph, pa = merged[-1]
        x, y, bw, bh, area = box
        gap = x - (px + pw)
        overlap = max(0, min(py + ph, y + bh) - max(py, y))
        if gap <= 2 and overlap >= 0.45 * min(ph, bh):
            nx = min(px, x)
            ny = min(py, y)
            nr = max(px + pw, x + bw)
            nb = max(py + ph, y + bh)
            merged[-1] = [nx, ny, nr - nx, nb - ny, pa + area]
        else:
            merged.append(list(box))

    # Numeric values are right-aligned. If noise survives, the rightmost
    # digit-like components are the real value.
    merged = [tuple(b) for b in merged]
    if len(merged) > max_digits:
        merged = merged[-max_digits:]
    return merged


def normalize_digit(binary, box):
    x, y, bw, bh, _ = box
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + bw + pad)
    y2 = min(binary.shape[0], y + bh + pad)
    glyph = binary[y1:y2, x1:x2]
    ys, xs = np.where(glyph > 0)
    if len(xs) == 0:
        return None, None

    glyph = glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    gh, gw = glyph.shape[:2]
    if gh < 8 or gw < 2:
        return None, None

    scale = min(20.0 / max(1, gw), 20.0 / max(1, gh))
    nw = max(1, int(round(gw * scale)))
    nh = max(1, int(round(gh * scale)))
    resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    xoff = (28 - nw) // 2
    yoff = (28 - nh) // 2
    canvas[yoff : yoff + nh, xoff : xoff + nw] = resized
    return canvas, glyph


def hole_features(glyph):
    # Count only sizeable enclosed regions. JPEG/block noise can create tiny
    # child contours that previously turned a real LCD 0 into a false 8.
    def analyse(img):
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]
        min_area = max(4.0, 0.018 * float(img.shape[0] * img.shape[1]))
        holes = []
        for idx, item in enumerate(hierarchy):
            if item[3] < 0:
                continue
            area = abs(cv2.contourArea(contours[idx]))
            if area < min_area:
                continue
            m = cv2.moments(contours[idx])
            if m["m00"] == 0:
                continue
            cy = float(m["m01"] / m["m00"]) / max(1.0, float(img.shape[0]))
            holes.append((area, cy))
        return holes

    original = analyse(glyph)
    eroded = analyse(cv2.erode(glyph, np.ones((2, 2), np.uint8), iterations=1))
    # A spurious thin bridge in a blurred 0 disappears after erosion. A real
    # 8 normally keeps two substantial lobes. Prefer the simpler topology.
    holes = eroded if 0 < len(eroded) < len(original) else original
    holes.sort(reverse=True)
    return len(holes), [round(cy, 3) for _, cy in holes]


class FontTemplates:
    def __init__(self):
        self.bank = {d: [] for d in range(10)}
        TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
        for digit in range(10):
            folder = TEMPLATE_ROOT / str(digit)
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.png"))[:12]:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape == (28, 28):
                    self.bank[digit].append(img)

    @staticmethod
    def _score(a, b):
        aa = cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32) / 255.0
        bb = cv2.GaussianBlur(b, (3, 3), 0).astype(np.float32) / 255.0
        best = 0.0
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                shifted = np.zeros_like(aa)
                ys = max(0, dy)
                ye = min(28, 28 + dy)
                xs = max(0, dx)
                xe = min(28, 28 + dx)
                src_ys = max(0, -dy)
                src_ye = src_ys + (ye - ys)
                src_xs = max(0, -dx)
                src_xe = src_xs + (xe - xs)
                shifted[ys:ye, xs:xe] = aa[src_ys:src_ye, src_xs:src_xe]
                denom = float(np.linalg.norm(shifted) * np.linalg.norm(bb))
                if denom > 1e-6:
                    best = max(best, float(np.sum(shifted * bb) / denom))
        return best

    def classify(self, norm):
        best_digit = None
        best_score = 0.0
        second_score = 0.0
        for digit, templates in self.bank.items():
            digit_best = 0.0
            for templ in templates:
                digit_best = max(digit_best, self._score(norm, templ))
            if digit_best > best_score:
                second_score = best_score
                best_score = digit_best
                best_digit = digit
            elif digit_best > second_score:
                second_score = digit_best
        margin = best_score - second_score
        return best_digit, best_score, margin

    def learn(self, digit, norm):
        digit = int(digit)
        if digit < 0 or digit > 9:
            return False
        existing = self.bank[digit]
        if existing:
            same = max(self._score(norm, x) for x in existing)
            if same >= 0.985:
                return False
        if len(existing) >= 12:
            return False
        folder = TEMPLATE_ROOT / str(digit)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{int(time.time() * 1000)}.png"
        if not cv2.imwrite(str(path), norm):
            return False
        existing.append(norm.copy())
        return True

    def counts(self):
        return {str(d): len(v) for d, v in self.bank.items() if v}


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values -= np.max(values)
    exp = np.exp(values)
    total = np.sum(exp)
    if not math.isfinite(total) or total <= 0:
        return np.zeros_like(values)
    return exp / total


class DigitModel:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def classify(self, digit, holes=None, hole_centers=None):
        variants = [
            digit,
            cv2.dilate(digit, np.ones((2, 2), np.uint8), iterations=1),
            cv2.erode(digit, np.ones((2, 2), np.uint8), iterations=1),
        ]
        probs = []
        for variant in variants:
            tensor = variant.astype(np.float32)[None, None, :, :] / 255.0
            logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
            probs.append(softmax(np.asarray(logits).reshape(-1)))
        mean_prob = np.mean(np.stack(probs, axis=0), axis=0)

        # Aqotec's own LCD topology is a stronger cue than MNIST for looped
        # digits. This is intentionally font-specific: 0 has one central hole,
        # 8 has two, 9 one high hole and 6 one low hole.
        ranking = list(np.argsort(mean_prob)[::-1])
        topology_digit = None
        topology_conf = 0.0
        centers = hole_centers or []
        if holes == 2:
            topology_digit, topology_conf = 8, 0.94
        elif holes == 1 and centers:
            cy = centers[0]
            if 0.38 <= cy <= 0.62:
                topology_digit, topology_conf = 0, 0.94
            elif cy < 0.38:
                topology_digit, topology_conf = 9, 0.88
            elif cy > 0.62:
                topology_digit, topology_conf = 6, 0.88

        if topology_digit is not None:
            digit_value = topology_digit
            confidence = max(float(mean_prob[digit_value]), topology_conf)
        else:
            digit_value = int(ranking[0])
            confidence = float(mean_prob[digit_value])

        top3 = [(int(d), round(float(mean_prob[d]) * 100.0, 1)) for d in ranking[:3]]
        return digit_value, confidence, top3


def extract_glyphs(crop, max_digits=3):
    binary = make_binary(crop)
    boxes = digit_boxes(binary, max_digits)
    items = []
    for box in boxes:
        norm, glyph = normalize_digit(binary, box)
        if norm is not None and glyph is not None:
            items.append((norm, glyph, box))
    return binary, items


def recognize_integer(model, templates, crop, max_digits=3):
    binary, items = extract_glyphs(crop, max_digits)
    if not items:
        return None, 0.0, [], binary, [], []

    digits = []
    confidences = []
    diagnostics = []
    norms = []
    for norm, glyph, box in items:
        norms.append(norm)
        template_digit, template_score, template_margin = templates.classify(norm)
        holes, hole_centers = hole_features(glyph)

        if (
            template_digit is not None
            and template_score >= 0.88
            and template_margin >= 0.035
        ):
            value = int(template_digit)
            confidence = min(0.99, max(0.88, template_score))
            top3 = []
            source = "aqotec_template"
        else:
            value, confidence, top3 = model.classify(
                norm, holes=holes, hole_centers=hole_centers
            )
            source = "onnx"

        digits.append(str(value))
        confidences.append(confidence)
        diagnostics.append(
            {
                "box": [int(v) for v in box[:4]],
                "holes": holes,
                "hole_centers": hole_centers,
                "digit": value,
                "confidence": round(confidence * 100.0, 1),
                "source": source,
                "template_digit": template_digit,
                "template_score": round(template_score * 100.0, 1),
                "template_margin": round(template_margin * 100.0, 1),
                "top3": top3,
            }
        )

    if not digits:
        return None, 0.0, [], binary, diagnostics, norms

    value = int("".join(digits))
    confidence = float(min(confidences))
    return value, confidence, confidences, binary, diagnostics, norms


def expected_digits(value, decimals=0):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if decimals == 0:
        return str(int(round(abs(value))))
    scaled = int(round(abs(value) * (10 ** decimals)))
    return str(scaled)


def calibrate_templates(screen, templates, values):
    specs = [
        ("energy", values.get("energy"), 0, 6),
        ("vorlauf", values.get("vorlauf"), 1, 4),
        ("ruecklauf", values.get("ruecklauf"), 1, 4),
        ("spread", values.get("spread"), 1, 3),
    ]
    learned = []
    for name, raw_value, decimals, max_digits in specs:
        label = expected_digits(raw_value, decimals)
        if not label:
            continue
        crop = crop_rect(screen, CALIBRATION_WINDOWS[name])
        _, items = extract_glyphs(crop, max_digits)
        if len(items) != len(label):
            continue
        for char, (norm, _, _) in zip(label, items):
            if templates.learn(int(char), norm):
                learned.append(int(char))

    # Strong zero bootstrap: when the existing independent path says both
    # power and flow are zero, the same live LCD provides two zero examples.
    if values.get("power") == 0.0 and values.get("flow") == 0.0:
        for name in ("power", "flow"):
            crop = field_crop(screen, name)
            _, items = extract_glyphs(crop, 2)
            if len(items) == 1:
                if templates.learn(0, items[0][0]):
                    learned.append(0)
    return learned


class Stability:
    def __init__(self):
        self.last = None
        self.count = 0

    def update(self, value, confidence, threshold, required):
        if value is None or confidence < threshold:
            self.count = 0
            self.last = None
            return False
        if value == self.last:
            self.count += 1
        else:
            self.last = value
            self.count = 1
        return self.count >= required


def save_debug(name, crop, binary):
    root = Path("/data/debug")
    root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    cv2.imwrite(str(root / f"{stamp}_{name}_crop.jpg"), crop)
    cv2.imwrite(str(root / f"{stamp}_{name}_binary.png"), binary)
    files = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[40:]:
        try:
            path.unlink()
        except OSError:
            pass


def main():
    if not TOKEN:
        raise RuntimeError("SUPERVISOR_TOKEN_missing")

    options = read_options()
    templates = FontTemplates()
    model = DigitModel()
    stability = {"power": Stability(), "flow": Stability()}
    publish_discovery()

    log(
        "started",
        camera=options["camera_entity"],
        scan_interval=options["scan_interval"],
        model="MNIST-12 ONNX + learned Aqotec font templates v0.1.3",
        mode="shadow_only",
    )

    while True:
        started = time.monotonic()
        payload = {
            "source": "aqotec-digitocr-shadow-v1.3",
            "status": "starting",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            image = fetch_camera(options["camera_entity"])
            quad = fixed_screen_quad(image)
            screen_mode = "fixed_calibrated"
            screen_conf = 1.0
            screen = warp_screen(image, quad)

            live_values = {
                "energy": fetch_float_state("input_number.aqotec_energie"),
                "vorlauf": fetch_float_state("input_number.aqotec_vorlauf"),
                "ruecklauf": fetch_float_state("input_number.aqotec_ruecklauf"),
                "spread": fetch_float_state("input_number.aqotec_spreizung"),
                "power": fetch_float_state("input_number.aqotec_leistung"),
                "flow": fetch_float_state("input_number.aqotec_durchfluss"),
            }
            learned = calibrate_templates(screen, templates, live_values)

            results = {}
            for field in ("power", "flow"):
                crop = field_crop(screen, field)
                max_digits = 2 if field == "power" else 4
                value, confidence, per_digit_conf, binary, diagnostics, norms = recognize_integer(
                    model, templates, crop, max_digits
                )
                results[field] = {
                    "value": value,
                    "confidence": confidence,
                    "digit_confidences": per_digit_conf,
                    "diagnostics": diagnostics,
                    "norms": norms,
                }
                if options["save_debug_crops"]:
                    save_debug(field, crop, binary)

            rapid_power = live_values.get("power")
            rapid_flow = live_values.get("flow")

            power = results["power"]["value"]
            flow = results["flow"]["value"]
            power_conf = results["power"]["confidence"]
            flow_conf = results["flow"]["confidence"]

            power_stable = stability["power"].update(
                power, power_conf, options["min_digit_confidence"], options["stable_samples"]
            )
            flow_stable = stability["flow"].update(
                flow, flow_conf, options["min_digit_confidence"], options["stable_samples"]
            )

            status = "ok"
            if power is None or flow is None:
                status = "segmentation_error"
            elif power_conf < options["min_digit_confidence"] or flow_conf < options["min_digit_confidence"]:
                status = "low_confidence"
            elif screen_mode == "fixed_fallback":
                status = "ok_fixed_geometry"

            payload.update(
                {
                    "status": status,
                    "power": power,
                    "flow": flow,
                    "power_confidence": round(power_conf * 100.0, 1),
                    "flow_confidence": round(flow_conf * 100.0, 1),
                    "power_digit_confidences": [round(x * 100.0, 1) for x in results["power"]["digit_confidences"]],
                    "flow_digit_confidences": [round(x * 100.0, 1) for x in results["flow"]["digit_confidences"]],
                    "power_diagnostics": results["power"]["diagnostics"],
                    "flow_diagnostics": results["flow"]["diagnostics"],
                    "power_stable": power_stable,
                    "flow_stable": flow_stable,
                    "rapidocr_power": rapid_power,
                    "rapidocr_flow": rapid_flow,
                    "power_delta": None if power is None or rapid_power is None else round(power - rapid_power, 1),
                    "flow_delta": None if flow is None or rapid_flow is None else round(flow - rapid_flow, 1),
                    "power_agrees": None if power is None or rapid_power is None else int(power) == int(round(rapid_power)),
                    "flow_agrees": None if flow is None or rapid_flow is None else int(flow) == int(round(rapid_flow)),
                    "screen_mode": screen_mode,
                    "screen_confidence": screen_conf,
                    "model": "mnist-12.onnx + learned Aqotec font templates",
                    "template_counts": templates.counts(),
                    "templates_learned_this_scan": learned,
                    "shadow_only": True,
                }
            )
        except Exception as exc:
            payload.update({"status": "error", "error": str(exc)})
            log("scan_error", error=str(exc))

        try:
            mqtt_publish("aqotec/digitocr/state", payload, retain=False)
        except Exception as exc:
            log("mqtt_error", error=str(exc))

        elapsed = time.monotonic() - started
        log(
            "scan",
            status=payload.get("status"),
            power=payload.get("power"),
            flow=payload.get("flow"),
            power_confidence=payload.get("power_confidence"),
            flow_confidence=payload.get("flow_confidence"),
            power_diag=payload.get("power_diagnostics"),
            flow_diag=payload.get("flow_diagnostics"),
            template_counts=payload.get("template_counts"),
            learned=payload.get("templates_learned_this_scan"),
            screen_mode=payload.get("screen_mode"),
            elapsed_s=round(elapsed, 3),
        )
        time.sleep(max(1.0, options["scan_interval"] - elapsed))


if __name__ == "__main__":
    main()

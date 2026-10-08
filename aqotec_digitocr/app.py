import base64
import zlib
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
# One source of truth: rectified Aqotec power and flow number rows.
# Camera fixed in position on 2026-10-08. No automatic moving windows.
FIELD_WINDOWS = {
    "power": (0.755, 0.375, 0.940, 0.505),
    "flow": (0.755, 0.490, 0.940, 0.600),
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
    }
    try:
        if OPTIONS_PATH.exists():
            defaults.update(json.loads(OPTIONS_PATH.read_text(encoding="utf-8")))
    except Exception as exc:
        log("options_error", error=str(exc))
    defaults["scan_interval"] = max(20, min(600, int(defaults["scan_interval"])))
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
        "sw_version": "0.1.13",
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
            # Locked Aqotec LCD geometry from the stationary camera
            # recalibration on 2026-10-08 (perspective quadrilateral).
            [0.382 * w, 0.065 * h],
            [0.958 * w, 0.105 * h],
            [0.968 * w, 0.695 * h],
            [0.363 * w, 0.653 * h],
        ],
        dtype=np.float32,
    )


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
    if foreground_ratio > 0.45:
        return np.zeros_like(binary)

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

    # Do not reject legitimate glyphs at the artificial preprocessing
    # border. Reliability is enforced by multi-sample and hydraulic checks.

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


# Two fixed zero-glyph samples from the user's Aqotec camera (2026-10-08).
# No automatic training, image storage or survey recording.
ZERO_TEMPLATES = {
    "power": "c-qa<F%AGA2m`?O|DSdsy0s-TXh+W|3cxpEW$Fw}uVP2XU5$J7))CGx?>>O;d%T}gba#wHI$Eypj~zntFd=!u#{M7+=~w{s",
    "flow": "c-qa<F%AGA2m`?O|DSfX1`6sz*CR$kff3=6%K75^WZ?Qz;>?}$kI?Cz-q~|_mu<VQdmG26e;xRuLB4`jNvjqYjyYM&#Q|3",
}


def recognize_zero(binary, boxes, field):
    if len(boxes) != 1:
        return False, 0.0
    x, y, w, h, _ = boxes[0]
    # Only a complete, right-aligned single digit may match the zero reference.
    if not (22 <= w <= 58 and 30 <= h <= 90):
        return False, 0.0
    glyph = binary[y:y+h, x:x+w]
    height, width = glyph.shape
    scale = min(24.0 / width, 24.0 / height)
    w2, h2 = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(glyph, (w2, h2), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    dx, dy = (32 - w2) // 2, (32 - h2) // 2
    canvas[dy:dy+h2, dx:dx+w2] = resized
    reference = np.frombuffer(zlib.decompress(base64.b85decode(ZERO_TEMPLATES[field])), np.uint8).reshape(32, 32)
    score = float(cv2.matchTemplate(canvas, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score >= 0.86, score


class Stability:
    def __init__(self):
        self.candidate = None
        self.count = 0
        self.accepted = None
        self.accepted_confidence = 0.0

    def update(self, value, confidence, threshold, required):
        if value is None or confidence < threshold:
            self.count = 0
            self.candidate = None
            return self.accepted, False

        if value == self.candidate:
            self.count += 1
        else:
            self.candidate = value
            self.count = 1

        newly_accepted = False
        if self.count >= required:
            if self.accepted != value:
                newly_accepted = True
            self.accepted = value
            self.accepted_confidence = confidence
        return self.accepted, newly_accepted


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
        model="MNIST-12 ONNX + learned Aqotec font templates v0.1.13",
        mode="shadow_only",
    )

    while True:
        started = time.monotonic()
        payload = {
            "source": "aqotec-digitocr-shadow-v1.13",
            "status": "starting",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            image = fetch_camera(options["camera_entity"])
            screen = warp_screen(image, fixed_screen_quad(image))
            # Fail closed if the blue display is absent or camera moved away.
            roi = screen[120:500, 120:780]
            blue_minus_red = np.median(
                roi[:, :, 0].astype(np.int16) - roi[:, :, 2].astype(np.int16)
            )
            if blue_minus_red < 45:
                raise RuntimeError("aqotec_display_not_visible")
            screen_mode, screen_conf = "fixed", 1.0

            live_values = {
                "power": fetch_float_state("input_number.aqotec_leistung"),
                "flow": fetch_float_state("input_number.aqotec_durchfluss"),
            }
            results = {}
            for field in ("power", "flow"):
                crop = crop_rect(screen, FIELD_WINDOWS[field])
                max_digits = 2 if field == "power" else 4
                value, confidence, per_digit_conf, binary, diagnostics, norms = recognize_integer(
                    model, templates, crop, max_digits
                )
                # Verify the obvious on-screen zero before trusting generic ONNX.
                boxes = digit_boxes(binary, max_digits)
                is_zero, zero_score = recognize_zero(binary, boxes, field)
                if is_zero:
                    value, confidence = 0, max(0.90, zero_score)
                    per_digit_conf = [confidence]
                    diagnostics = [{
                        "box": [int(z) for z in boxes[0][:4]],
                        "digit": 0,
                        "source": "fixed_zero_reference",
                        "confidence": round(confidence * 100.0, 1),
                    }]

                # Require the last glyph to end at the Aqotec numeric
                # right-alignment column, not at a cropped label/unit.
                right_edges = [
                    FIELD_WINDOWS[field][0] * WARP_W + d["box"][0] + d["box"][2]
                    for d in diagnostics
                ]
                right_edge = max(right_edges) if right_edges else None
                # Digits end around x=0.925 in the fixed screen image.
                valid_alignment = (
                    right_edge is not None
                    and 0.88 * WARP_W <= right_edge <= 0.955 * WARP_W
                )
                if not valid_alignment:
                    value, confidence = None, 0.0
                results[field] = {
                    "value": value,
                    "confidence": confidence,
                    "alignment_valid": valid_alignment,
                    "right_edge": round(right_edge, 1) if right_edge is not None else None,
                    "digit_confidences": per_digit_conf,
                    "diagnostics": diagnostics,
                    "norms": norms,
                }

            rapid_power = live_values.get("power")
            rapid_flow = live_values.get("flow")

            power = results["power"]["value"]
            flow = results["flow"]["value"]
            power_conf = results["power"]["confidence"]
            flow_conf = results["flow"]["confidence"]

            # A hot-water loop cannot deliver kW from an almost zero L/h flow.
            # 0.10 kW per L/h assumes an intentionally broad, conservative
            # maximum temperature difference (about 86 K for water).
            physically_impossible = (
                power is not None and flow is not None and
                power > max(0.25, 0.10 * flow)
            )
            if physically_impossible:
                power, flow = None, None
            safe_confidence = 0.75
            safe_samples = 3
            accepted_power, power_changed = stability["power"].update(
                power, power_conf, safe_confidence, safe_samples
            )
            accepted_flow, flow_changed = stability["flow"].update(
                flow, flow_conf, safe_confidence, safe_samples
            )
            power_stable = accepted_power is not None and power == accepted_power
            flow_stable = accepted_flow is not None and flow == accepted_flow

            status = "ok"
            if physically_impossible:
                status = "held_physical_inconsistency"
            elif power is None or flow is None:
                status = "held_segmentation_error"
            elif accepted_power is None or accepted_flow is None:
                status = "warming_up"
            elif power_conf < safe_confidence or flow_conf < safe_confidence:
                status = "held_low_confidence"
            elif not power_stable or not flow_stable:
                status = "held_unstable"

            # Publish neither stale held values nor unverified measurements.
            # Do not fabricate zeros when the camera/segmentation fails.
            if status != "ok":
                accepted_power, accepted_flow = None, None
            payload.update(
                {
                    "status": status,
                    "power": accepted_power,
                    "flow": accepted_flow,
                    "raw_power": power,
                    "raw_flow": flow,
                    "power_confidence": round(power_conf * 100.0, 1),
                    "flow_confidence": round(flow_conf * 100.0, 1),
                    "power_stable": power_stable,
                    "flow_stable": flow_stable,
                    "power_changed": power_changed,
                    "flow_changed": flow_changed,
                    "rapidocr_power": rapid_power,
                    "rapidocr_flow": rapid_flow,
                    "power_delta": None if accepted_power is None or rapid_power is None else round(accepted_power - rapid_power, 1),
                    "flow_delta": None if accepted_flow is None or rapid_flow is None else round(accepted_flow - rapid_flow, 1),
                    "power_agrees": None if accepted_power is None or rapid_power is None else int(accepted_power) == int(round(rapid_power)),
                    "flow_agrees": None if accepted_flow is None or rapid_flow is None else int(accepted_flow) == int(round(rapid_flow)),
                    "screen_mode": "fixed",
                    "physical_plausibility_pass": not physically_impossible,
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
            raw_power=payload.get("raw_power"),
            raw_flow=payload.get("raw_flow"),
            elapsed_s=round(elapsed, 2),
        )
        time.sleep(max(1.0, options["scan_interval"] - elapsed))


if __name__ == "__main__":
    main()

# Changelog

## 0.1.1
- Switch digit preprocessing from grayscale to the red channel for stronger white-glyph separation on the blue Aqotec LCD.
- Replace projection-only segmentation with connected-component digit boxes.
- Add LCD hole-topology correction for recurring 0/8 confusion.
- Add per-digit boxes, hole counts and top-3 classifier diagnostics.


## 0.1.0
- Initial shadow-only release.
- Independent camera read path.
- Automatic screen detection with fixed calibrated fallback.
- Digit-only ONNX classification for power and flow.
- MQTT Discovery sensors with confidence and RapidOCR comparison values.
- No writes to the existing Aqotec measurement/control path.

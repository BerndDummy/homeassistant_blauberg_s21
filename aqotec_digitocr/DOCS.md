# Aqotec DigitOCR Shadow

## Purpose

This app runs beside the existing Aqotec RapidOCR app. It is a shadow measurement path only.

- Existing RapidOCR remains untouched.
- Existing Aqotec helpers and automations remain untouched.
- DigitOCR reads only the camera image.
- DigitOCR publishes only new entities prefixed with **Aqotec DigitOCR**.
- No images are stored unless `save_debug_crops` is explicitly enabled.
- Debug storage is hard-limited to 40 small crop/binary files.

## Model

The first baseline uses the MIT-licensed ONNX Model Zoo MNIST-12 model as a digit-only classifier. Each recognized display digit is segmented from the fixed numeric fields and classified independently.

The app additionally publishes per-field confidence and direct deltas against the existing RapidOCR raw values so the two pipelines can be evaluated side by side before any future promotion.

## Default entities

- Aqotec DigitOCR Leistung
- Aqotec DigitOCR Durchfluss
- Aqotec DigitOCR Leistung Konfidenz
- Aqotec DigitOCR Durchfluss Konfidenz
- Aqotec DigitOCR vs RapidOCR Leistung
- Aqotec DigitOCR vs RapidOCR Durchfluss
- Aqotec DigitOCR Status

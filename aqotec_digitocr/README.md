# Aqotec DigitOCR Shadow

A strictly parallel, read-only shadow OCR for the Aqotec display.

It reads **power** and **flow** from the existing Aqotec camera with a digit-only ONNX classifier and publishes separate Home Assistant MQTT Discovery entities.

It never writes to the existing Aqotec input_numbers, automations, heating control or RapidOCR pipeline.

This directory contains the reference ONNX artifact used by `neuro-onnx-state-classifier`.

The checked-in model is a tiny linear classifier with a softmax output over:

- `quiescent`
- `subthreshold`
- `spiking`

To regenerate it, run:

```bash
python scripts/generate_reference_model.py
```

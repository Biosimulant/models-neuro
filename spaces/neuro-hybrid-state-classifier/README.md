# Neuro: Hybrid State Classifier

## Scientific Question
Can a compact ONNX model classify mechanistic Hodgkin-Huxley state into coarse behavioral regimes without replacing the underlying mechanistic simulation?

## Hybrid Design
- `current_source` drives the mechanistic neuron population.
- `neuron` remains the causal simulator of voltage and gating dynamics.
- `state_adapter` compresses the mechanistic state payload into a fixed feature vector.
- `state_classifier` consumes that vector with an ONNX model and emits class probabilities.
- `state_monitor` preserves the full mechanistic trace for interpretation.

## Why This Space Exists
This is the reference hybrid pattern for BioSimulant:
- mechanistic model produces the biology
- adapter translates outputs into ML-friendly features
- ONNX model adds a learned inference layer

## Expected Behaviors
- Clear mechanistic membrane-voltage traces from `state_monitor`
- Stable feature-vector output from `state_adapter`
- ONNX probabilities over `quiescent`, `subthreshold`, and `spiking`

## Known Limitations
- The ONNX artifact is intentionally tiny and illustrative.
- Classification quality is not the point; the compositional pattern is.
- Only one mechanistic population is included in the baseline reference space.

## How to Run
```bash
python spaces/neuro-hybrid-state-classifier/run_local.py --duration auto --tick-dt auto
python spaces/neuro-hybrid-state-classifier/simui_local.py --port 8765
```

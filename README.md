# models-neuro

Curated collection of **custom-built neuroscience** simulation models for the **biosim** platform. This repository contains hand-crafted Python implementations of spiking neural networks, synaptic dynamics, neural monitoring tools, and input generators — designed for composable, YAML-configurable neural simulations.

## What's Inside

### Models (10 packages)

Each model is a custom Python implementation designed for modular composition.

**Neuroscience** — spiking neural networks, synaptic dynamics, and neural monitoring:

| Model | Description |
|-------|-------------|
| `neuro-izhikevich-population` | Spiking neuron population (Regular Spiking, Fast Spiking presets) |
| `neuro-hodgkin-huxley-population` | Conductance-based Hodgkin-Huxley neuron population |
| `neuro-hodgkin-huxley-state-monitor` | Detailed HH state monitor (V, gates, ionic currents) |
| `neuro-exp-synapse-current` | Exponential-decay synapses with configurable connectivity |
| `neuro-step-current` | Constant/step current injection into neurons |
| `neuro-poisson-input` | Poisson-distributed spike train generator |
| `neuro-spike-monitor` | Spike raster visualization |
| `neuro-rate-monitor` | Firing rate computation and display |
| `neuro-state-monitor` | Neuron state variable tracking (membrane potential, etc.) |
| `neuro-spike-metrics` | Summary statistics from spike streams |

## How It Works

These are **native Python models**, not SBML imports. They implement the `biosim.BioModule` interface and are designed to be wired together via `space.yaml` for complex neural simulations without writing code.

### Example Wiring

```yaml
wiring:
  - from: current_source.current
    to: [neuron.input_current]
  - from: neuron.spikes
    to: [spike_monitor.spikes, rate_monitor.spikes]
```

## Getting Started

### Prerequisites
- Python 3.11+
- `biosim` framework

```bash
pip install "biosim @ git+https://github.com/BioSimulant/biosim.git@main"
```

### Create Neural Circuits

These models are building blocks for complex neural simulations. See the **neuroscience-**** repositories for SBML/CellML/NeuroML models from literature.

## License

Dual-licensed: Apache-2.0 (code), CC BY 4.0 (content)

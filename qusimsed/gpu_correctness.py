"""GPU-only numerical validation using a fixed PennyLane VQC workload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gpu import lightning_gpu_device
from .validation import validate_training


def run(qubits: int = 4, layers: int = 2, seed: int = 7, learning_rate: float = 0.05) -> dict:
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:
        raise RuntimeError("GPU correctness run requires PennyLane") from exc
    device, environment = lightning_gpu_device(qubits)
    parameters = layers * qubits * 3
    rng = np.random.default_rng(seed)
    initial = pnp.array(rng.normal(0, 0.1, parameters), requires_grad=True)
    features = pnp.array(rng.normal(size=qubits), requires_grad=False)

    @qml.qnode(device, interface="autograd", diff_method="parameter-shift")
    def circuit(theta):
        for wire, feature in enumerate(features): qml.RY(feature, wires=wire)
        index = 0
        for _ in range(layers):
            for wire in range(qubits):
                qml.RX(theta[index], wires=wire); qml.RY(theta[index + 1], wires=wire); qml.RZ(theta[index + 2], wires=wire); index += 3
            for wire in range(qubits): qml.CNOT(wires=[wire, (wire + 1) % qubits])
        return qml.expval(qml.PauliZ(0))

    # The reference is conventional PennyLane parameter-shift. The candidate
    # manually schedules the identical +/- shift evaluations, preserving order
    # when it aggregates them; it is a correctness check, not a speed claim.
    reference_gradient = qml.grad(circuit)(initial)
    shift = np.pi / 2; values = []
    for index in range(parameters):
        plus = initial.copy(); minus = initial.copy(); plus[index] += shift; minus[index] -= shift
        values.append((circuit(plus) - circuit(minus)) / 2)
    candidate_gradient = pnp.array(values)
    reference_expectation = circuit(initial); candidate_expectation = circuit(initial.copy())
    reference_loss = (reference_expectation - 1.0) ** 2; candidate_loss = (candidate_expectation - 1.0) ** 2
    reference_updated = initial - learning_rate * reference_gradient; candidate_updated = initial - learning_rate * candidate_gradient
    payload = validate_training({"expectation": reference_expectation, "gradient": reference_gradient, "loss": reference_loss, "parameters": reference_updated},
                                {"expectation": candidate_expectation, "gradient": candidate_gradient, "loss": candidate_loss, "parameters": candidate_updated})
    return {"environment": environment.__dict__, "configuration": {"qubits": qubits, "layers": layers, "seed": seed, "learning_rate": learning_rate, "differentiation": "parameter-shift"}, "correctness": payload}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--qubits", type=int, default=4); parser.add_argument("--layers", type=int, default=2); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--output", default="results/gpu-correctness.json")
    args = parser.parse_args(); result = run(args.qubits, args.layers, args.seed)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

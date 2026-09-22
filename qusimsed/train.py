"""Run application training and keep a localhost Streamlit dashboard available."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from .config import ExperimentConfig
from .real_benchmarks import run_real_benchmark


class Dashboard:
    def __init__(self, output_dir, port=8501):
        self.output_dir = Path(output_dir).resolve()
        self.port = port
        self.process = None
        self.log = None

    def start(self):
        # Do not mistake an unrelated service's health endpoint for our process.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.port))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment['QUSIMSED_OUTPUT_DIR'] = str(self.output_dir)
        environment['PYTHONPATH'] = str(root) + os.pathsep + environment.get('PYTHONPATH', '')
        self.log = (self.output_dir / 'streamlit.log').open('a', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'streamlit', 'run', str(root / 'app/streamlit_app.py'),
                 '--server.address', '127.0.0.1', '--server.port', str(self.port),
                 '--global.developmentMode', 'false',
                 '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                cwd=root, env=environment, stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Streamlit exited. Check {self.output_dir / "streamlit.log"}; install streamlit in this Python environment.')
                try:
                    with urlopen(f'http://127.0.0.1:{self.port}/_stcore/health', timeout=.5) as response:
                        if response.status == 200:
                            return self
                except OSError:
                    pass
                time.sleep(.2)
            raise RuntimeError('Streamlit did not become ready within 30 seconds')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['iris', 'wine', 'breast-cancer', 'mnist-pca'], default='iris')
    parser.add_argument('--qubits', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--parameters', type=int)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--differentiation', choices=['parameter-shift', 'adjoint'], default='adjoint')
    parser.add_argument('--backend', choices=['torch-cuda', 'torch-cpu', 'pennylane'], default='torch-cuda')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--sm-demand', type=float, default=None)
    parser.add_argument('--partition-size', type=int, default=16)
    parser.add_argument('--affinity-weight', type=float, default=1.)
    parser.add_argument('--sync-weight', type=float, default=1.)
    parser.add_argument('--parallelism-weight', type=float, default=1.)
    parser.add_argument('--resource-weight', type=float, default=1.)
    parser.add_argument('--memory-gib', type=float)
    parser.add_argument('--memory-safety-factor', type=float, default=.8)
    parser.add_argument('--save-traces', action='store_true')
    parser.add_argument('--output-dir', default='results/training')
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--no-ui', action='store_true', help='Train without starting Streamlit')
    parser.add_argument('--exit-after-training', action='store_true', help='Stop the dashboard when training ends')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    config = ExperimentConfig(workload=args.dataset, qubits=args.qubits, layers=args.layers,
                              trainable_parameters=args.parameters, iterations=args.epochs,
                              samples=args.samples, learning_rate=args.learning_rate, seed=args.seed,
                              differentiation=args.differentiation, execution_backend=args.backend,
                              gpu_device=args.gpu_device, streams=args.streams, sm_demand=args.sm_demand,
                              partition_size=args.partition_size, affinity_weight=args.affinity_weight,
                              sync_weight=args.sync_weight, parallelism_weight=args.parallelism_weight, resource_weight=args.resource_weight, memory_safety_factor=args.memory_safety_factor,
                              memory_budget_bytes=None if args.memory_gib is None else int(args.memory_gib * (1 << 30)),
                              enable_trace=args.save_traces, output_dir=str(Path(args.output_dir).resolve()))
    config.validate()
    dashboard = None
    try:
        if not args.no_ui:
            dashboard = Dashboard(config.output_dir, args.port).start()
            print(f'Dashboard ready: http://127.0.0.1:{args.port}\n'
                  f'On your laptop: ssh -N -L {args.port}:127.0.0.1:{args.port} USER@SERVER\n'
                  f'Then open http://localhost:{args.port}', flush=True)
        print('Training started. Results are saved automatically.', flush=True)
        result = run_real_benchmark(config)
        print(json.dumps({'status': result['status'], 'collection': result['collection'], 'rows': result['rows']}, indent=2), flush=True)
        if dashboard and not args.exit_after_training:
            print('Training finished. Dashboard remains available; refresh Saved results. Press Ctrl+C to stop.', flush=True)
            while dashboard.process.poll() is None:
                time.sleep(.5)
    except KeyboardInterrupt:
        print('Stopping training/dashboard.', flush=True)
    finally:
        if dashboard:
            dashboard.stop()


if __name__ == '__main__':
    main()

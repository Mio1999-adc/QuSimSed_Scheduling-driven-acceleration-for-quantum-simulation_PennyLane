"""Automatic, per-invocation experiment archives; writes stay outside timings."""
from functools import wraps
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import traceback
import uuid


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, default=_json_default) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def save_archive(result, directory):
    """Also refresh the archive after a caller attaches profiling metadata."""
    directory = Path(directory)
    temporary = directory / 'result.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, default=_json_default), encoding='utf-8')
    temporary.replace(directory / 'result.json')
    metadata = result.get('configuration', {})
    base = {key: metadata.get(key) for key in ('qubits', 'layers', 'differentiation', 'seed', 'workload', 'execution_backend')}
    base['parameters'] = metadata.get('trainable_parameters') or 3 * metadata.get('qubits', 0) * metadata.get('layers', 0)
    rows = [{**base, **row} for row in result.get('rows', [])]
    _table(directory / 'summary.csv', rows)
    _table(directory / 'errors.csv', [
        {**base, 'method': row['method'], **{key: value for key, value in row.items()
         if any(token in key for token in ('error', 'max_abs', 'max_rel', 'rmse', 'mean_abs', 'tolerance', 'test_loss'))}}
        for row in result.get('rows', [])])
    _table(directory / 'timings.csv', [
        {**base, 'method': row['method'], 'sample_index': index, 'elapsed_ms': elapsed}
        for row in result.get('rows', []) for index, elapsed in enumerate(row.get('iteration_times_ms', []))])
    _table(directory / 'convergence.csv', [
        {**base, 'method': method, 'epoch': index + 1, 'training_loss': loss,
         'epoch_seconds': result.get('epoch_times_seconds', {}).get(method, [None] * len(losses))[index]}
        for method, losses in result.get('trajectories', {}).items() for index, loss in enumerate(losses)])


def collect_results(kind):
    """Archive normal correctness, benchmark and training calls automatically."""
    def decorate(function):
        @wraps(function)
        def wrapped(config, *args, **kwargs):
            if not config.collect_results:
                return function(config, *args, **kwargs)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            directory = (config.results_path / 'experiments' / f'{stamp}-{kind}-{uuid.uuid4().hex[:8]}').resolve()
            directory.mkdir(parents=True, exist_ok=False)
            record = {'configuration': config.metadata(), 'run_kind': kind, 'status': 'running',
                      'collection': {'directory': str(directory), 'result': str(directory / 'result.json')}}
            save_archive(record, directory)
            try:
                result = function(config, *args, **kwargs)
            except BaseException as exc:
                record.update(status='failed', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                save_archive(record, directory)
                raise
            result.update(collection=record['collection'], run_kind=kind)
            valid = all(row.get('all_within_tolerance', row.get('correctness_within_tolerance', True))
                        for row in result.get('rows', []))
            result['status'] = 'completed' if valid else 'validation_failed'
            result['validation_scope'] = {'correctness': 'independent numerical reference',
                                          'training': 'trajectory and parameters versus Sequential',
                                          'benchmark': 'timing only; no independent correctness check in this call'}[kind]
            save_archive(result, directory)
            return result
        return wrapped
    return decorate

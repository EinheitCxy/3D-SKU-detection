"""Independent bounded CPU diagnostic; never overlap the formal video runs."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def main():
    path = Path(__file__).resolve().parent / 'point-range-probe.json'
    if path.exists():
        raise FileExistsError(path)
    if np.__version__ != '1.26.4':
        raise RuntimeError('This probe is registered for the measured NumPy1.26.4 environment')
    labels = np.arange(2_500_000, dtype=np.int32)
    labels //= 1000
    queries = np.arange(200, dtype=np.int32) * 14
    results, timings = {}, {}
    for mode in ('python_int', 'typed_scalar', 'typed_vector'):
        start = perf_counter()
        if mode == 'typed_vector':
            output = np.column_stack((np.searchsorted(labels, queries, side='left'),
                                      np.searchsorted(labels, queries, side='right')))
        else:
            output = np.asarray([
                (np.searchsorted(labels, int(query) if mode == 'python_int' else query, side='left'),
                 np.searchsorted(labels, int(query) if mode == 'python_int' else query, side='right'))
                for query in queries])
        timings[mode] = perf_counter() - start
        results[mode] = output
    equal = all(np.array_equal(results['python_int'], results[mode]) for mode in ('typed_scalar','typed_vector'))
    if not equal:
        raise AssertionError('Different range endpoints in the diagnostic')
    result = dict(status='complete', numpy=np.__version__, labels_count=len(labels), queries_count=len(queries),
                  labels_dtype=str(labels.dtype), native_long_bytes=np.dtype(np.int_).itemsize,
                  timings_s=timings, range_endpoints_equal=equal,
                  scope='One independent synthetic CPU diagnostic after all video runs; not production data or end-to-end evidence; not included in cohort times; no production change')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

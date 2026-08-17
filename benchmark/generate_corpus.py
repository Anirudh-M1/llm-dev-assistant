"""Generate a synthetic corpus of small Python files for the ingestion benchmark.

The real corpus/ directory only has 3 files and 4 functions -- far too small
to show a meaningful difference between sequential and concurrent embedding
requests. This generates a larger set of trivial-but-valid Python functions
purely as a load-bearing fixture for benchmark/run_ingest_benchmark.py, not
as a stand-in for a real codebase.

Run: python -m benchmark.generate_corpus
"""

import os

OPS = ["+", "-", "*"]

FUNCTION_TEMPLATE = '''def op_{i}_{j}(a, b):
    """Synthetic benchmark function {i}-{j}."""
    return a {op} b
'''


def generate(out_dir, num_files=60, functions_per_file=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_files):
        functions = [
            FUNCTION_TEMPLATE.format(i=i, j=j, op=OPS[(i + j) % len(OPS)])
            for j in range(functions_per_file)
        ]
        path = os.path.join(out_dir, f"module_{i}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(functions))


if __name__ == "__main__":
    generate(os.path.join(os.path.dirname(__file__), "corpus"))

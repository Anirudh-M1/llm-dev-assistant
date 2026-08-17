"""Generate N separate synthetic "repos" for the multi-repo latency benchmark.

Each repo is its own directory with a handful of files/functions, so the
latency benchmark can ingest each one under a distinct repo_id and issue
concurrent queries across genuinely separate FAISS indices -- not a single
shared corpus split at query time.

Run: python -m benchmark.generate_multi_repo_corpus
"""

import os

OPS = ["+", "-", "*", "//"]

FUNCTION_TEMPLATE = '''def op_{i}_{j}(a, b):
    """Synthetic benchmark function {i}-{j}."""
    return a {op} b
'''


def generate(out_dir, num_repos=12, files_per_repo=4, functions_per_file=5):
    for repo_i in range(num_repos):
        repo_dir = os.path.join(out_dir, f"repo_{repo_i}")
        os.makedirs(repo_dir, exist_ok=True)
        for file_i in range(files_per_repo):
            functions = [
                FUNCTION_TEMPLATE.format(
                    i=file_i, j=func_i, op=OPS[(repo_i + file_i + func_i) % len(OPS)]
                )
                for func_i in range(functions_per_file)
            ]
            path = os.path.join(repo_dir, f"module_{file_i}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(functions))


if __name__ == "__main__":
    generate(os.path.join(os.path.dirname(__file__), "multi_repo_corpus"))

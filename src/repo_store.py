"""In-memory per-repo chunk + FAISS index storage.

Keyed by repo_id so multiple codebases can be ingested and queried
independently -- and, from an HTTP client's perspective, concurrently.
"""


class RepoStore:
    def __init__(self):
        self._repos = {}

    def put(self, repo_id, chunks, index):
        self._repos[repo_id] = {"chunks": chunks, "index": index}

    def get(self, repo_id):
        return self._repos.get(repo_id)

    def repo_ids(self):
        return list(self._repos.keys())

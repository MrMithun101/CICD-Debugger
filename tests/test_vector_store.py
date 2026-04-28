"""
Tests for the vector store layer.

Unit tests mock the ChromaDB collection to verify interface behaviour.
The integration test uses EphemeralClient + a dummy embedding function so
it runs offline without downloading the sentence-transformer model.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from chromadb import EmbeddingFunction, Documents, Embeddings

from ingestion.models import ProcessedChunk
from vector_store.chroma_store import ChromaStore, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(i: int = 0, failure_type: str = "test_failure") -> ProcessedChunk:
    return ProcessedChunk(
        chunk_id=f"owner__repo__1__{i}",
        chunk_index=i,
        source_step=f"job/step_{i}.txt",
        text=f"FAILED test_example_{i} - AssertionError",
        repo="owner/repo",
        workflow_name="CI",
        run_id=1,
        failure_type=failure_type,
        created_at="2024-06-01T00:00:00",
        html_url="https://github.com/owner/repo/actions/runs/1",
    )


class _DummyEF(EmbeddingFunction):
    """Embedding function that returns distinct unit vectors — no model download required.

    Each document gets a unique vector based on its index so that cosine similarity
    calculations don't degenerate on zero vectors.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for i, _ in enumerate(input):
            vec = [0.0] * 384
            vec[i % 384] = 1.0
            embeddings.append(vec)
        return embeddings


@pytest.fixture
def mock_store(tmp_path: Path) -> ChromaStore:
    """ChromaStore with a mocked PersistentClient — no disk I/O, no model download."""
    with patch("vector_store.chroma_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        # Inject DummyEF so __init__ never calls get_embedding_function()
        store = ChromaStore(db_path=tmp_path, embedding_function=_DummyEF())
        store.collection = mock_collection
        return store


@pytest.fixture
def ephemeral_store() -> ChromaStore:
    """ChromaStore backed by an in-memory ChromaDB — integration tests only.

    Calls client.reset() before each test because EphemeralClient uses a
    module-level shared system that persists state across test functions.
    """
    client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
    client.reset()  # guarantee clean state regardless of test ordering
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_DummyEF(),
        metadata={"hnsw:space": "cosine"},
    )
    store = ChromaStore.__new__(ChromaStore)
    store.client = client
    store.collection = collection
    return store


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_calls_collection_with_correct_ids(self, mock_store: ChromaStore):
        chunks = [_make_chunk(0), _make_chunk(1)]
        mock_store.upsert(chunks)
        call_kwargs = mock_store.collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["owner__repo__1__0", "owner__repo__1__1"]

    def test_calls_collection_with_correct_documents(self, mock_store: ChromaStore):
        chunk = _make_chunk(0)
        mock_store.upsert([chunk])
        call_kwargs = mock_store.collection.upsert.call_args.kwargs
        assert call_kwargs["documents"] == [chunk.text]

    def test_metadata_contains_required_fields(self, mock_store: ChromaStore):
        chunk = _make_chunk(0, failure_type="build_error")
        mock_store.upsert([chunk])
        meta = mock_store.collection.upsert.call_args.kwargs["metadatas"][0]
        assert meta["repo"] == "owner/repo"
        assert meta["failure_type"] == "build_error"
        assert meta["run_id"] == 1
        assert meta["chunk_index"] == 0

    def test_empty_list_is_noop(self, mock_store: ChromaStore):
        mock_store.upsert([])
        mock_store.collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def _chroma_result(self, n: int = 2) -> dict:
        """Build the nested-list structure ChromaDB actually returns."""
        return {
            "ids": [[f"chunk_{i}" for i in range(n)]],
            "documents": [[f"log text {i}" for i in range(n)]],
            "distances": [[0.1 * i for i in range(n)]],
            "metadatas": [[
                {
                    "repo": "owner/repo",
                    "workflow_name": "CI",
                    "run_id": 1,
                    "failure_type": "test_failure",
                    "source_step": f"job/step_{i}.txt",
                    "created_at": "2024-06-01T00:00:00",
                    "html_url": "https://github.com",
                    "chunk_index": i,
                }
                for i in range(n)
            ]],
        }

    def test_returns_empty_when_store_is_empty(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 0
        result = mock_store.query("some error")
        assert result == []
        mock_store.collection.query.assert_not_called()

    def test_flattens_chroma_nested_results(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 2
        mock_store.collection.query.return_value = self._chroma_result(2)
        results = mock_store.query("test error", n_results=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk_0"
        assert results[1]["chunk_id"] == "chunk_1"

    def test_result_includes_distance_field(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 1
        mock_store.collection.query.return_value = self._chroma_result(1)
        results = mock_store.query("error")
        assert "distance" in results[0]
        assert isinstance(results[0]["distance"], float)

    def test_result_merges_metadata_fields(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 1
        mock_store.collection.query.return_value = self._chroma_result(1)
        result = mock_store.query("error")[0]
        assert result["repo"] == "owner/repo"
        assert result["failure_type"] == "test_failure"

    def test_failure_type_filter_sets_where_clause(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 5
        mock_store.collection.query.return_value = self._chroma_result(1)
        mock_store.query("error", failure_type="build_error")
        call_kwargs = mock_store.collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"failure_type": "build_error"}

    def test_no_failure_type_filter_sends_no_where_clause(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 5
        mock_store.collection.query.return_value = self._chroma_result(1)
        mock_store.query("error", failure_type=None)
        call_kwargs = mock_store.collection.query.call_args.kwargs
        assert call_kwargs["where"] is None

    def test_n_results_clamped_to_collection_size(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 3
        mock_store.collection.query.return_value = self._chroma_result(1)
        mock_store.query("error", n_results=100)
        call_kwargs = mock_store.collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 3


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_delegates_to_collection(self, mock_store: ChromaStore):
        mock_store.collection.count.return_value = 42
        assert mock_store.count() == 42


# ---------------------------------------------------------------------------
# Integration — EphemeralClient, no model download
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_upsert_and_query_round_trip(self, ephemeral_store: ChromaStore):
        chunks = [_make_chunk(i) for i in range(3)]
        ephemeral_store.upsert(chunks)

        assert ephemeral_store.count() == 3

        results = ephemeral_store.query("AssertionError", n_results=2)
        assert len(results) == 2
        assert all("chunk_id" in r for r in results)
        assert all("distance" in r for r in results)
        assert all("repo" in r for r in results)

    def test_upsert_is_idempotent(self, ephemeral_store: ChromaStore):
        chunk = _make_chunk(0)
        ephemeral_store.upsert([chunk])
        ephemeral_store.upsert([chunk])  # same chunk_id — should overwrite, not duplicate
        assert ephemeral_store.count() == 1

    def test_failure_type_filter_restricts_results(self, ephemeral_store: ChromaStore):
        ephemeral_store.upsert([
            _make_chunk(0, failure_type="test_failure"),
            _make_chunk(1, failure_type="build_error"),
            _make_chunk(2, failure_type="build_error"),
        ])
        results = ephemeral_store.query("error", n_results=5, failure_type="build_error")
        assert all(r["failure_type"] == "build_error" for r in results)

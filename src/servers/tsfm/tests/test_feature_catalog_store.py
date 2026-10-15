"""Feature catalog store tests.

These cover the database-backed store logic through the in-memory test double, so
they do not require a live database.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from servers.tsfm import main as tsfm_main
from servers.tsfm.core.store import MemoryStore
from servers.tsfm.stores import feature_store

from .conftest import call_tool


_CATALOG = (
    Path(__file__).resolve().parents[3]
    / "couchdb"
    / "scenarios_data"
    / "shared"
    / "tsfm"
    / "feature_catalog.json"
)


def _feature_doc(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    out.setdefault("_id", f"feature:{out['feature_id']}")
    return out


def seeded_feature_store() -> MemoryStore:
    store = MemoryStore()
    with _CATALOG.open() as fh:
        for doc in json.load(fh):
            store.put(feature_store.collection_name(), _feature_doc(doc))
    return store

def test_feature_store_reads_collection_name_from_env(monkeypatch):
    monkeypatch.setenv("FEATURE_CATALOG_DBNAME", "custom_feature_catalog")
    store = MemoryStore()
    with _CATALOG.open() as fh:
        doc = _feature_doc(json.load(fh)[0])
    store.put("custom_feature_catalog", doc)

    assert feature_store.collection_name() == "custom_feature_catalog"
    assert feature_store.get_feature(store, doc["feature_id"])["feature_id"] == doc[
        "feature_id"
    ]
    assert feature_store.find_features(store, kind="extractor")


def test_register_feature_never_executes_code():
    store = MemoryStore()
    bad = {
        "feature_id": "bad_inplace",
        "interface": "fit_transform",
        "code": (
            "class Transformation:\n"
            "    def fit(self, X, metadata):\n"
            "        return {}\n"
            "    def transform(self, X, state):\n"
            "        X[:] = 0\n"
            "        return X\n"
        ),
    }

    with pytest.raises(Exception, match="disabled"):
        feature_store.register_feature(store, bad)

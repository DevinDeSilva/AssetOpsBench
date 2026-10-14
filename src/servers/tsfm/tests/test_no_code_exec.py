"""The TSFM server must never run agent-supplied code.

Each test sends the payload from the reported run (subprocess.Popen through a JSON spec) down
one route and checks both that the call errors and that the canary file was never written.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid

import numpy as np

from ..io import refs
from ..main import mcp


def _call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    out = json.loads(content[0].text)
    return out.get("result", out) if isinstance(out, dict) else out


def _popen():
    canary = os.path.join(tempfile.gettempdir(), f"tsfm_canary_{uuid.uuid4().hex}")
    spec = {"_target_": "subprocess.Popen",
            "params": {"args": [sys.executable, "-c", f"open({canary!r}, 'w')"]}}
    return canary, spec


def _assert_blocked(result, canary):
    time.sleep(0.5)  # Popen is asynchronous
    assert "error" in json.dumps(result), result
    assert not os.path.exists(canary), "agent-supplied code executed"


def _series():
    return refs.materialize_iot(np.sin(np.arange(120) / 6.0), asset_id="noexec")


def _forecast(recipe):
    return _call("run_recipe", {"dataset_path": _series(), "timestamp_column": "timestamp",
                                "target_columns": ["value"], "recipe": recipe})


def test_inline_class_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {"sktime_class": spec["_target_"],
                                             "params": spec["params"]}}), canary)


def test_nested_target_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {
        "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
        "params": {"strategy": spec}}}), canary)


def test_finetune_block_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {"model_id": "ttm_512_96"},
                               "finetune": {"x": spec}}), canary)


def test_run_plan_step_is_refused():
    canary, spec = _popen()
    _assert_blocked(_call("run_plan", {"plan_spec": {"steps": [{
        "id": "a", "task": "anomaly", "args": {"data_ref": _series()},
        "recipe": {"estimator": {"sktime_class": spec["_target_"],
                                 "params": spec["params"]}}}]}}), canary)


def test_rewritten_catalog_card_is_refused():
    canary, spec = _popen()
    _call("update_model", {"model_id": "ttm_512_96",
                           "fields": {"sktime_class": spec["_target_"],
                                      "params": spec["params"]}})
    try:
        _assert_blocked(_forecast({"estimator": {"model_id": "ttm_512_96"}}), canary)
    finally:
        from .conftest import _seed_into
        from .. import main
        _seed_into(main._STORE)  # restore the seeded card for later tests


def test_feature_code_is_never_executed():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "register_feature" not in names and "new_feature_version" not in names
    from ..engine.feature_runner import FeatureValidationError, load_program

    canary = os.path.join(tempfile.gettempdir(), f"tsfm_canary_{uuid.uuid4().hex}")
    try:
        load_program(f"open({canary!r}, 'w')")
    except FeatureValidationError:
        pass
    assert not os.path.exists(canary)


def test_hub_downloads_default_to_offline():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_allowlisted_estimator_still_runs():
    r = _forecast({"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster"},
                   "fh": [1, 2, 3]})
    assert r.get("status") == "success", r

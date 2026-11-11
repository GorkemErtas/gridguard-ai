import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_validation_only_dashboard_stops_before_loading_predictions(tmp_path):
    artifacts = tmp_path / "validation"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(
        json.dumps({"test": None, "validation": {"random_forest": {"average_precision": 0.2}}})
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "streamlit_app.py"))
    app.run()
    app.sidebar.text_input[0].set_value(str(artifacts)).run()
    assert not app.exception
    assert "validation-only" in app.info[0].value

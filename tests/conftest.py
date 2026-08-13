from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def real_registry():
    from app import app

    return app.extensions["model_registry"]


@pytest.fixture(scope="session")
def real_client(real_registry):
    from app import create_app

    application = create_app({"TESTING": True}, registry=real_registry)
    return application.test_client()

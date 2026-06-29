import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


import urllib.request


@pytest.fixture(scope="session")
def test_pdf_path() -> Path:
    """Generate a small PDF for testing."""
    from tests.test_pdf import create_test_pdf
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    create_test_pdf(tmp, pages=5)
    return tmp


@pytest.fixture(scope="session")
def academic_pdf_path() -> Path:
    """Download a real academic PDF for integration testing."""
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    urllib.request.urlretrieve(url, tmp)
    assert tmp.stat().st_size > 100000
    return tmp

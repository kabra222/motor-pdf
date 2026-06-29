import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_pdf_path() -> Path:
    """Generate a small PDF for testing."""
    from tests.test_pdf import create_test_pdf
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    create_test_pdf(tmp, pages=5)
    return tmp

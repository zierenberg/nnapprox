import importlib
import sys
import pytest

from nnapprox.core.exceptions import BackendNotAvailableError

def test_passing_example():
    assert True

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
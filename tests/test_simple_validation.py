"""Simple validation test to verify pytest infrastructure."""

import pytest
from pathlib import Path


def test_basic_assertion():
    """Test that basic assertions work."""
    assert 1 + 1 == 2
    assert "hello" + " world" == "hello world"
    assert len([1, 2, 3]) == 3


def test_fixtures_work(temp_dir, mock_config):
    """Test that our basic fixtures work."""
    # Test temp_dir fixture
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    
    # Test mock_config fixture
    assert isinstance(mock_config, dict)
    assert "batch_size" in mock_config
    assert mock_config["learning_rate"] == 0.001


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works."""
    import time
    # Don't actually sleep in validation
    assert True


class TestPytestFeatures:
    """Test class-based tests work."""
    
    def test_in_class(self):
        """Test method in class."""
        assert 2 * 2 == 4
    
    def test_with_fixture(self, temp_dir):
        """Test using fixture in class."""
        assert temp_dir.is_dir()


@pytest.mark.parametrize("x,y,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (10, -5, 5),
])
def test_parametrized(x, y, expected):
    """Test parametrized tests work."""
    assert x + y == expected
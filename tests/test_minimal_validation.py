"""Minimal validation test to check pytest works without dependencies."""

import pytest


def test_basic_assertion():
    """Test that basic assertions work."""
    assert 1 + 1 == 2


def test_pytest_features():
    """Test basic pytest features."""
    with pytest.raises(ValueError):
        raise ValueError("Expected error")


@pytest.mark.unit
def test_marker():
    """Test that markers work."""
    assert True
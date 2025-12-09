import doctest
import nnapprox

def test_doctests():
    results = doctest.testmod(nnapprox)
    assert results.failed == 0
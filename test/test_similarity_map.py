from . import items, smap
import random


def test_no_associations():
    """Test number of associations of a map."""
    assert smap.no_associations() == 4, f"Expected 4, got {smap.no_associations()}"


def test_association_sizes():
    """Test number of similarities used by each association."""
    expected = [1, 2, 2, 1]
    assert (
        smap.association_sizes() == expected
    ), f"Expected {expected}, got {smap.association_sizes()}"


def test_association_offsets():
    """Test association offsets."""
    expected = [0, 1, 3, 5]
    assert (
        smap.association_offsets() == expected
    ), f"Expected {expected}, got {smap.association_offsets()}"


def test_similarity_map_length():
    """Test similarity map length."""
    expected = len(items)
    assert len(smap.lcols) == expected, f"Expected {expected}, got {len(smap.lcols)}"


def test_similarity_map_iter():
    """Test similarity map iteration."""
    assert list(smap) == items, f"Expected: \n{items},\n got {list(smap)}"


def test_similarity_map_item():
    """Test similarity map item."""
    i = random.sample(range(len(items)), 1)[0]
    expected = items[i]
    assert smap[i] == expected, f"Expected {expected}, got {smap[i]}"

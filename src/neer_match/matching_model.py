"""
Matching models module.

This module provides a common import interface for different matching model
implementations.
"""

from neer_match.dl_matching_model import DLMatchingModel
from neer_match.ns_matching_model import NSMatchingModel

assert DLMatchingModel is not None
assert NSMatchingModel is not None

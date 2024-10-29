"""Auxiliary test variables."""

from neer_match.examples import games
from neer_match.similarity_encoding import SimilarityEncoder
from neer_match.similarity_map import SimilarityMap

instructions = {
    "title": ["jaro_winkler"],
    "platform": ["levenshtein", "jaro"],
    "year": ["euclidean", "discrete"],
    "developer~dev": ["jaro"],
}

items = [
    ("title", "title", "jaro_winkler"),
    ("platform", "platform", "levenshtein"),
    ("platform", "platform", "jaro"),
    ("year", "year", "euclidean"),
    ("year", "year", "discrete"),
    ("developer", "dev", "jaro"),
]

smap = SimilarityMap(instructions)

sencoder = SimilarityEncoder(smap)

left = games.left
right = games.right
matches = games.matches

"""Auxiliary test variables."""

from neer_match.examples import games
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
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

left_short = left.iloc[0:3, :]
right_short = right.iloc[0:3, :]
matches_short = matches.join(left_short, on="left", how="inner")[
    ["left", "right"]
].join(right_short, on="right", how="inner")[["left", "right"]]

dl_model = DLMatchingModel(smap)
ns_model = NSMatchingModel(smap)

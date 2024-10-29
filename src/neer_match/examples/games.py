"""Fuzzy Games Matching."""

from importlib.resources import files
import pandas

left = pandas.read_csv(
    files("neer_match.examples.data").joinpath("fuzzy-games-left.csv")
)
right = pandas.read_csv(
    files("neer_match.examples.data").joinpath("fuzzy-games-right.csv")
)
matches = pandas.read_csv(
    files("neer_match.examples.data").joinpath("fuzzy-games-matches.csv")
)

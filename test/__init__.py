"""Auxiliary test variables."""

from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_encoding import SimilarityEncoder
from neer_match.similarity_map import SimilarityMap
import pandas as pd
import random

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

left = left = pd.DataFrame(
    {
        "title": [
            # fmt: off
            "Metal Gear Solid 4: Guns of the Patriots", "Metal Gear Solid",
            "Metal Gear Solid 3: Snake Eater", "Metal Gear Solid V: The Phantom Pain",
            "Metal Gear Solid Mobile", "Metal Gear Solid: Portable Ops",
            "Metal Gear Solid 2: Substance", "Metal Gear Solid: The Twin Snakes",
            "Metal Gear Solid", "Metal Gear Rising: Revengeance",
            "Metal Gear Solid Integral", "Metal Gear Solid: Peace Walker HD Edition",
            "Metal Slug 4 Mobile", "Metal Gear Rising: Revengeance",
            "Metal Gear Online", "Metal Gear Acid",
            "Metal Slug Anthology", "Metal Gear Solid V: Ground Zeroes",
            "Metal Slug Mobile", "Metal Gear Rising: Revengeance - Jetstream",
            "Metal Slug 2", "Metal Gear Solid V: Ground Zeroes",
            "Metal Slug 7", "Metal Gear Solid: VR Missions",
            "Metal Slug 4 & 5", "Metal Gear Rising: Revengeance - Blade Wolf",
            "Metal Gear Acid 2", "Metal Slug XX",
            "Metal Slug Defense", "Metal Slug 1",
            "Metal Gear Solid V: Metal Gear Online", "Metal Slug Touch",
            "Metal Slug X", "Metal Gear Rising: Revengeance - Jetstream",
            "Metal Gear Survive", "Metal Gear",
            # fmt: on
        ],
        "platform": [
            # fmt: off
            "PS3", "PS", "PS2", "XONE", "MOBI", "PSP", "XBOX", "GC", "PC", "PC", "PS",
            "PS3", "MOBI", "X360", "PS3", "PSP", "PS4", "PS4", "MOBI", "PS3", "IOS",
            "XONE", "DS", "PS", "XBOX", "X360", "MOBI", "PSP", "IOS", "IOS", "PS4",
            "IOS", "IOS", "X360", "PS4", "MOBI",
            # fmt: on
        ],
        "year": [
            # fmt: off
            2008, 1998, 2004, 2015, 2008, 2006, 2002, 2004, 2000, 2014, 1999, 2012,
            2008, 2013, 2008, 2005, 2016, 2014, 2004, 2013, 2013, 2014, 2008, 1999,
            2005, 2013, 2009, 2010, 2014, 2012, 2015, 2009, 2013, 2013, 2018, 2004
            # fmt: on
        ],
        "scores": [
            # fmt: off
            93.53, 93.24, 91.77, 90.38, 92.50, 86.95, 86.66, 85.58, 84.22, 83.55, 90.00,
            90.00, 90.00, 82.56, 80.37, 76.70, 80.00, 75.23, 75.00, 75.00, 77.00, 72.50,
            72.11, 70.64, 70.47, 70.00, 70.00, 68.97, 70.00, 68.33, 67.50, 60.00, 64.00,
            60.00, 61.09, 50.00
            # fmt: on
        ],
        "reviews": [
            # fmt: off
            85, 29, 86, 12, 2, 60, 56, 69, 18, 11, 1, 1, 1, 31, 8, 52, 1, 48, 4, 2, 5,
            6, 42, 18, 33, 2, 1, 18, 6, 3, 2, 1, 5, 1, 33, 1
            # fmt: on
        ],
        "developer": [
            # fmt: off
            "Kojima Productions/Konami", "KCEJ/Konami", "KCEJ/Konami",
            "Kojima Productions/Konami", "Ideaworks3D/Konami",
            "Kojima Productions/Konami", "KCEJ/Konami", "Silicon Knights/Konami",
            "Digital Dialect/Konami", "PlatinumGames/Konami", "Konami",
            "Kojima Productions/Konami", "SNK Playmore/I-Play", "PlatinumGames/Konami",
            "Kojima Productions/Konami", "Konami", "Terminal Reality/SNK Playmore",
            "Kojima Productions/Konami", "SNK Playmore/I-Play", "PlatinumGames/Konami",
            "SNK Playmore", "Kojima Productions/Konami",
            "SNK Playmore/Ignition Entertainment", "KCEJ/Konami",
            "BrezzaSoft/SNK Playmore", "PlatinumGames/Konami", "Konami/Glu Mobile",
            "SNK Playmore", "SNK Playmore", "SNK Playmore", "Kojima Productions/Konami",
            "SNK Playmore", "SNK Playmore", "PlatinumGames/Konami", "Konami",
            "Konami Mobile & Online, Inc./Konami"
            # fmt: on
        ],
    }
)

right = left.copy()
no_duplicates = 3
right = pd.concat([right, right.sample(no_duplicates)])

tpos = right.columns.get_loc("title")
dpos = right.columns.get_loc("developer")

for r in range(right.shape[0]):
    no_chars = random.randint(1, 3)
    title = right.iloc[r, tpos]
    right.iloc[r, tpos] = "".join(
        [
            l
            for i, l in enumerate(title)
            if i not in random.sample(range(len(title)), no_chars)
        ]
    )
    no_chars = random.randint(1, 2)
    developer = right.iloc[r, dpos]
    right.iloc[r, dpos] = "".join(
        [
            l
            for i, l in enumerate(developer)
            if i not in random.sample(range(len(developer)), no_chars)
        ]
    )

right.rename(columns={"developer": "dev"}, inplace=True)

matches = pd.DataFrame(
    {
        "left": left.index.append(right.iloc[-no_duplicates:,].index),
        "right": range(len(right)),
    }
)

left_short = left.iloc[0:3, :]
right_short = right.iloc[0:3, :]
matches_short = matches.join(left_short, on="left", how="inner")[
    ["left", "right"]
].join(right_short, on="right", how="inner")[["left", "right"]]

dl_model = DLMatchingModel(smap)
ns_model = NSMatchingModel(smap)

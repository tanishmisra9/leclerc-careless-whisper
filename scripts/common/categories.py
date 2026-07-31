"""Shared category -> search query definitions used by all crawl_*.py scripts."""

# Wikimedia Commons full-text search matches literally, so long qualifier-heavy phrases
# (e.g. "face closeup", "SF-24 car 16") often return zero hits even when relevant images
# exist. Use shorter, title-like phrasing for this source instead of CATEGORY_QUERIES.
WIKIMEDIA_CATEGORY_QUERIES = {
    "face": ["Charles Leclerc", "Charles Leclerc portrait", "Charles Leclerc interview"],
    "helmet": ["Charles Leclerc helmet", "Charles Leclerc onboard"],
    "car_positive": ["Charles Leclerc Ferrari", "Ferrari SF-24"],
    "car_negative": ["Lewis Hamilton Ferrari", "Ferrari F1 car", "Red Bull F1 car"],
    "background_negative": ["F1 pit crew", "F1 grid walk", "Max Verstappen", "Lando Norris"],
}

CATEGORY_QUERIES = {
    "face": [
        "Charles Leclerc face closeup",
        "Charles Leclerc interview",
        "Charles Leclerc press conference",
    ],
    "helmet": [
        "Charles Leclerc helmet",
        "Charles Leclerc onboard camera",
        "Charles Leclerc grid walk",
    ],
    "car_positive": [
        "Charles Leclerc Ferrari SF-24 car 16",
        "Charles Leclerc Ferrari onboard",
        "Charles Leclerc Ferrari F1 race",
    ],
    "car_negative": [
        "Lewis Hamilton Ferrari car 44",
        "Ferrari F1 car empty track",
        "Red Bull F1 car onboard",
        "Mercedes F1 car onboard",
    ],
    "background_negative": [
        "F1 grid walk crowd",
        "F1 pit crew",
        "F1 timing tower graphics",
        "Max Verstappen interview",
        "Lando Norris interview",
    ],
}

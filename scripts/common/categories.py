"""Category -> search query definitions used by crawl_*.py scripts.

v1 scope is face-only (see project handoff plan). helmet/car_positive/car_negative/
background_negative were dropped for now — revisit when those tiers are built.
"""

CATEGORY_QUERIES = {
    "face": [
        "Charles Leclerc face closeup",
        "Charles Leclerc interview",
        "Charles Leclerc press conference",
    ],
}

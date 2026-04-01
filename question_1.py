import os, warnings
import pandas as pd
import numpy as np
import altair as alt

from conf import SIMPSONS_COLOR_SCHEME

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ─── colour palette (Okabe-Ito, colour-blind safe) ────────────────────────────

PRIMARY = SIMPSONS_COLOR_SCHEME["primary"]
SECONDARY = SIMPSONS_COLOR_SCHEME["secondary"]
TERTIARY = SIMPSONS_COLOR_SCHEME["tertiary"]

primary_color = alt.value(PRIMARY)
secondary_color = alt.value(SECONDARY)
tertiary_color = alt.value(TERTIARY)

TITLE_FONT = 14
AXIS_FONT  = 11

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────

eps   = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
eps["original_air_date"] = pd.to_datetime(eps["original_air_date"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEASON AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

season = (
    eps.groupby(["season"])
    .agg(
        avg_rating   = ("imdb_rating",            "mean"),
        min_rating   = ("imdb_rating",            "min"),
        max_rating   = ("imdb_rating",            "max"),
        std_rating   = ("imdb_rating",            "std"),
        avg_viewers  = ("us_viewers_in_millions", "mean"),
        min_viewers  = ("us_viewers_in_millions", "min"),
        max_viewers  = ("us_viewers_in_millions", "max"),
        std_viewers  = ("us_viewers_in_millions", "std"),
        n_episodes   = ("id",                     "count"),
        avg_votes    = ("imdb_votes",             "mean"),
        avg_views    = ("views",                  "mean"),
    )
    .round(3).reset_index()
)
# Label strings baked in Python — no tooltip needed
season["avg_rating_lbl"]  = season["avg_rating"].round(1).astype(str)
season["avg_viewers_lbl"] = season["avg_viewers"].round(1).astype(str)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLEAN SUBSETS
# ─────────────────────────────────────────────────────────────────────────────

eps_r  = eps.dropna(subset=["imdb_rating"]).copy()

# --- Chart Q1: Single Y-Axis with Clean Labels ---

# 1. Individual Episode Dots (Faint Background)
dots_r = (
    alt.Chart(eps_r)
    .mark_circle(size=15, opacity=0.33, color=TERTIARY)
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("imdb_rating:Q", title="IMDB Rating", scale=alt.Scale(domain=[4, 10])),
    )
)

# 2. 1-STD deviation band around the season average (as a shaded area)
band_r = (
    alt.Chart(season.dropna(subset=["avg_rating", "std_rating"]))
    .mark_area(opacity=0.25)
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("min_r:Q"),
        y2="max_r:Q",
        color=secondary_color,#era_color(legend=False),
    )
    .transform_calculate(
        min_r="datum.avg_rating - datum.std_rating",
        max_r="datum.avg_rating + datum.std_rating",
    )
)

# 3. Season Average Line and Fixed-Size Points
line_r = (
    alt.Chart(season.dropna(subset=["avg_rating"]))
    .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=50, filled=True))
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_rating:Q"),
        color=primary_color,#era_color(),
    )
)

# --- 4. DATA LABELS (Numbers above every dot) ---
lbl_dots_r = (
    alt.Chart(season.dropna(subset=["avg_rating"]))
    .mark_text(fontSize=10, dy=-15, color="#FFFFFF") 
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_rating:Q"),
        text=alt.Text("avg_rating:Q", format=".1f")
    )
)

# --- Combined Chart ---
chart_q1_clean = (
    # Ensure annot_pt_r is added at the end
    (dots_r + band_r + line_r + lbl_dots_r)
    .properties(
        width=850, 
        height=400,
        title=alt.TitleParams(
            "Q1 — Evolution of IMDB Ratings by Season",
            subtitle=[
                "Shaded Area: 1-STD Range · Faint Dots: Individual Episodes"
            ],
            fontSize=16, subtitleFontSize=11, subtitleColor="#ff82c1"
        )
    )
    .configure_view(stroke=None)
)

chart_q1_clean.save("outputs/q1_ratings_clean.html")

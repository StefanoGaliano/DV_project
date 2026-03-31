import os, warnings
import pandas as pd
import numpy as np
import altair as alt

from conf import HOMER_COLOR_SCHEME

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ─── colour palette (Okabe-Ito, colour-blind safe) ────────────────────────────
PRIMARY = HOMER_COLOR_SCHEME["primary"]
SECONDARY = HOMER_COLOR_SCHEME["secondary"]
TERTIARY = HOMER_COLOR_SCHEME["alternative_accent"]

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
    eps.groupby(["season"])#,"era"])
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
season["ci_rating"]  = (season["std_rating"]  / np.sqrt(season["n_episodes"]) * 1.96).round(3)
season["std_viewers"] = season["std_viewers"]
# Label strings baked in Python — no tooltip needed
season["avg_rating_lbl"]  = season["avg_rating"].round(1).astype(str)
season["avg_viewers_lbl"] = season["avg_viewers"].round(1).astype(str)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLEAN SUBSETS
# ─────────────────────────────────────────────────────────────────────────────

eps_r  = eps.dropna(subset=["imdb_rating"]).copy()
eps_v  = eps.dropna(subset=["us_viewers_in_millions"]).copy()


# ─── 2. STD BAND (Viewership) ────────────────────────────────────────────────
band_v = (
    alt.Chart(season.dropna(subset=["avg_viewers", "std_viewers"]))
    .mark_area(opacity=0.15)
    .encode(
        x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0, titleFontSize=AXIS_FONT)),
        y=alt.Y("min_v:Q", title="US Viewers (millions)", scale=alt.Scale(domain=[0, 35])),
        y2="max_v:Q",
        color=secondary_color,
    )
    .transform_calculate(
        min_v="datum.avg_viewers - datum.std_viewers",
        max_v="datum.avg_viewers + datum.std_viewers",
    )
)

# ─── 4. MIN-MAX RANGE RULES (Vertical whiskers) ─────────────────────────────
range_v = (
    alt.Chart(season.dropna(subset=["min_viewers", "max_viewers"]))
    .mark_rule(strokeWidth=1.2, opacity=0.4)
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("min_viewers:Q"),
        y2="max_viewers:Q",
        color=tertiary_color,
    )
)

# ─── 5. SEASON AVERAGE LINE + POINTS ────────────────────────────────────────
line_v = (
    alt.Chart(season.dropna(subset=["avg_viewers"]))
    .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=60, filled=True))
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_viewers:Q"),
        color=primary_color,
    )
)

# ─── 6. DATA LABELS (Consistent positioning above dots) ────────────────────
lbl_v = (
    alt.Chart(season.dropna(subset=["avg_viewers"]))
    .mark_text(fontSize=9, fontWeight="bold", dy=-12, color="#333333")
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_viewers:Q"),
        text=alt.Text("avg_viewers:Q", format=".1f")
    )
)


# ─── 8. COMBINED FINAL CHART ────────────────────────────────────────────────
chart_q2 = (
    # Layer order: Bars at the back, Labels/Annotations at the front
    alt.layer(band_v, range_v, line_v, lbl_v)#, locs_bar, era_annot_v)
    .properties(
        width=800, 
        height=350,
        title=alt.TitleParams(
            "Q2 — Evolution of US Viewership and Location Variety",
            subtitle=[
                "Numbers above dots = Avg Viewers (Millions) · Vertical Rules = Min-Max Range per Season",
                "Area = STD Range"
            ],
            fontSize=TITLE_FONT, subtitleFontSize=10, subtitleColor="#444",
        ),
    )
    .configure_view(stroke=None)
    .configure_legend(
        strokeColor='#DDD',
        cornerRadius=5,
        fillColor='white',
        padding=10
    )
)

chart_q2.save("outputs/q2_viewership_final.html")

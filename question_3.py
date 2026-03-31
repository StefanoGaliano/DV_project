import os
import warnings
import pandas as pd
import numpy as np
import altair as alt

from conf import SIMPSONS_COLOR_SCHEME
from util import gradient

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ─── colour palette (Okabe-Ito, colour-blind safe) ────────────────────────────
PRIMARY = SIMPSONS_COLOR_SCHEME["primary"]
SECONDARY = SIMPSONS_COLOR_SCHEME["secondary"]
TERTIARY = SIMPSONS_COLOR_SCHEME["tertiary"]

primary_color = alt.value(PRIMARY)
secondary_color = alt.value(SECONDARY)
tertiary_color = alt.value(TERTIARY)

SEASON_ORDER = list(range(1, 28))
SEASON_COLORS = gradient(PRIMARY, TERTIARY, len(SEASON_ORDER))

def season_color(legend=False):
    return alt.Color(
        "season:O", title="Season",
        scale=alt.Scale(domain=SEASON_ORDER, range=SEASON_COLORS),
        legend=alt.Legend(orient="top") if legend else None,
    )

TITLE_FONT = 14
AXIS_FONT  = 11

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────

eps   = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")

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
# Label strings baked in Python — no tooltip needed
season["avg_rating_lbl"]  = season["avg_rating"].round(1).astype(str)
season["avg_viewers_lbl"] = season["avg_viewers"].round(1).astype(str)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLEAN SUBSETS
# ─────────────────────────────────────────────────────────────────────────────

eps_r  = eps.dropna(subset=["imdb_rating"]).copy()
eps_v  = eps.dropna(subset=["us_viewers_in_millions"]).copy()
eps_rv = eps.dropna(subset=["imdb_rating","us_viewers_in_millions"]).copy()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  REGRESSION DATA  (Q3)
# ─────────────────────────────────────────────────────────────────────────────

x_vals = eps_rv["us_viewers_in_millions"].values
y_vals = eps_rv["imdb_rating"].values
coeffs = np.polyfit(x_vals, y_vals, 1)
x_range = np.linspace(x_vals.min(), x_vals.max(), 80)
reg_df = pd.DataFrame({
    "us_viewers_in_millions": x_range,
    "imdb_rating_fit": coeffs[0] * x_range + coeffs[1],
})
r_value = np.corrcoef(x_vals, y_vals)[0, 1]


# --- 2. Main Scatter Plot ---
scatter = (
    alt.Chart(eps_rv)
    .mark_point(size=60, opacity=0.8, filled=True)
    .encode(
        x=alt.X("us_viewers_in_millions:Q", title="US Viewers (millions)",
                axis=alt.Axis(titleFontSize=AXIS_FONT)),
        y=alt.Y("imdb_rating:Q", title="IMDB Rating",
                scale=alt.Scale(domain=[4, 10])),
        color=season_color(),
    )
)

# --- 4. Regression Line & Stats ---
reg_line = (
    alt.Chart(reg_df)
    .mark_line(color="#444", strokeWidth=2, strokeDash=[6, 4])
    .encode(
        x=alt.X("us_viewers_in_millions:Q"),
        y=alt.Y("imdb_rating_fit:Q"),
    )
)
# Regression equation label
reg_eq_df = pd.DataFrame([{
    "x": eps_rv["us_viewers_in_millions"].max() * 0.66,
    "y": 4.35,
    "label": f"r = {r_value:.2f}  ·  slope = {coeffs[0]:.3f}",
}])

reg_annot = (
    alt.Chart(pd.DataFrame([{"x": 22, "y": 4.2, "label": f"r = {r_value:.2f}"}]))
    .mark_text(fontSize=12, color="#FFFFFF", fontWeight="bold")
    .encode(x="x:Q", y="y:Q", text="label:N")
)

# --- 5. Combined Final Chart ---
chart_q3 = (
    (scatter + reg_line + reg_annot)
    .resolve_scale(color="independent", shape="independent") # Keep legends separate
    .properties(
        width=750, height=400,
        title=alt.TitleParams(
            "Q3 — Correlation: Quality vs. Popularity",
            subtitle=[
                "The dashed line shows the general downward trend as the show aged.",
                "Bluer points represent later seasons, while yellower points are from earlier seasons.",
            ],
            fontSize=TITLE_FONT, subtitleFontSize=10, subtitleColor="#FF82C1",
        ),
    )
    .configure_legend(
        fillColor="white",
        strokeColor="#DDD",
        cornerRadius=5,
        padding=10
    )
    .configure_view(stroke=None)
)

chart_q3.save("outputs/q3_correlation_final.html")

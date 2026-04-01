import os, warnings
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
ACCENT = SIMPSONS_COLOR_SCHEME["alternative_accent"]

primary_color = alt.value(PRIMARY)
secondary_color = alt.value(SECONDARY)
tertiary_color = alt.value(TERTIARY)
accent_color = alt.value(ACCENT)

TITLE_FONT = 14
AXIS_FONT  = 11


eps   = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
eps["original_air_date"] = pd.to_datetime(eps["original_air_date"])

# unique_viewer_counts = sorted(eps["us_viewers_in_millions"].unique().tolist())
# color_scale = alt.Scale(domain=unique_viewer_counts, range=gradient(PRIMARY, TERTIARY, len(unique_viewer_counts)))

hm_data = (
    eps[["season","number_in_season","us_viewers_in_millions",
         "imdb_rating", "title"]]
    .dropna(subset=["us_viewers_in_millions"])
    .copy()
)

heatmap = (
    alt.Chart(hm_data)
    .mark_rect()
    .encode(
        x=alt.X("number_in_season:O", title="Episode Number in Season",
                axis=alt.Axis(labelAngle=0, titleFontSize=AXIS_FONT)),
        y=alt.Y("season:O", title="Season",
                sort=alt.SortOrder("ascending"),
                axis=alt.Axis(titleFontSize=AXIS_FONT)),
        color=alt.Color(
            "us_viewers_in_millions:Q",
            title="Viewers (M)",
            scale=alt.Scale(domain=[2.5, 33.6], range=gradient(SIMPSONS_COLOR_SCHEME["accent"], SIMPSONS_COLOR_SCHEME["tertiary"], 5)),
            legend=None,
        ),
    )
)

# IMDB rating as text in each cell
hm_text = (
    alt.Chart(hm_data)
    .mark_text(fontSize=6.5)#, fontWeight="bold")
    .encode(
        x=alt.X("number_in_season:O"),
        y=alt.Y("season:O", sort=alt.SortOrder("ascending")),
        text=alt.Text("us_viewers_in_millions:Q", format=".1f"),
        color=alt.condition(
            alt.datum.us_viewers_in_millions > 15,
            alt.value("white"),
            alt.value("#424F46"),
        ),
    )
)




chart_q5 = (
    (heatmap + hm_text)
    .properties(
        width=760, height=400,
        title=alt.TitleParams(
            "Q5 — Season Viewer Patterns",
            subtitle=[
                "Cell colour = US viewers (pink, darker = more viewers) · "
                "Cell text = US viewers (millions)",
            ],
            fontSize=TITLE_FONT, subtitleFontSize=11, subtitleColor="#FF82C1",
        ),
    )
)


chart_q5.save("outputs/q5_final_heatmap.html")

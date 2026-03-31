import os
import warnings
import pandas as pd
import numpy as np
import altair as alt

from conf import HOMER_COLOR_SCHEME

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PRIMARY = HOMER_COLOR_SCHEME["primary"]
SECONDARY = HOMER_COLOR_SCHEME["secondary"]
TERTIARY = HOMER_COLOR_SCHEME["tertiary"]
ACCENT = HOMER_COLOR_SCHEME["alternative_accent"]

WEEKDAY_ORDER = ["Thursday", "Sunday"]
WEEKDAY_COLOURS = [PRIMARY, TERTIARY]
WEEKDAY_SCALE = alt.Scale(domain=WEEKDAY_ORDER, range=WEEKDAY_COLOURS)

TITLE_FONT = 14
AXIS_FONT  = 11

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────

eps   = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
eps["original_air_date"] = pd.to_datetime(eps["original_air_date"])

df = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")

# Drop episodes from other days of the week because there are too few to make a meaningful density plot
#df_thursday_sunday = df[df["air_dayofweek"].isin(["Thursday", "Sunday"])]

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
thu_sun = ["Thursday", "Sunday"]
day_order = weekdays

base = alt.Chart(df).properties(height=260, width=150)

violin = (
    base
    .transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=thu_sun))
    .transform_density(
        "us_viewers_in_millions",
        as_=["us_viewers_in_millions", "density"],
        extent=[0, df["us_viewers_in_millions"].max()],
        groupby=["air_dayofweek"],
    )
    .mark_area(orient="horizontal")
    .encode(
        x=alt.X("density:Q")
            .stack("center")
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0],ticks=False, grid=False),
        y=alt.Y("us_viewers_in_millions:Q", title="US Viewers (Millions)"),
        color=alt.Color("air_dayofweek:N", scale=WEEKDAY_SCALE, legend=None),
    )
)

# Rule plot for averages
avg = (
    base.
    transform_aggregate(mean_acc="mean(us_viewers_in_millions)", groupby=["air_dayofweek"])
    .transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=thu_sun))
    .mark_text(fontSize=15, fontWeight="bold", color=ACCENT)
    .encode(
        y=alt.Y("mean_acc:Q", title="US Viewers (Millions)"),
        color=alt.value(ACCENT),
        text=alt.Text("mean_acc:Q", format=".1f"),
    )
)


chart_q4 = (
    alt.layer(violin, avg)
    .facet(
        column=alt.Column(
            "air_dayofweek:N",
            sort=day_order,
            header=alt.Header(title=None, labelOrient="bottom", labelPadding=10),
        )
    )
    .transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=thu_sun))
    .configure_view(stroke=None)
    .properties(
        title=alt.TitleParams(
            "Q4 — Viewership Distribution by Airing Day",
            subtitle=[
                "Violion plot showing distribution of US viewers for episodes aired on Thursdays and Sundays.",
                "Numbers represent average viewership for each day."
                "Episodes aired on other days of the week are not shown due to insufficient data."
            ],
            fontSize=TITLE_FONT, subtitleFontSize=10, subtitleColor="#444",
        ),
    )
)

chart_q4.save("outputs/q4_weekday_viewers_violion_alt.html")

chart_q4.save("outputs/q4_viewers_by_day.html")

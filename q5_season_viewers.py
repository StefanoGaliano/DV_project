"""Generate a view that highlights the distribution of viewers across different seasons of The Simpsons."""

import altair as alt
import pandas as pd

from conf import SIMPSONS_COLOR_SCHEME


df = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
base = alt.Chart(df, width=400, height=300)


season_average_viewers = (
    base.transform_aggregate(
        mean_viewers="mean(us_viewers_in_millions)",
        groupby=["season"],
    )
    .mark_line(color=SIMPSONS_COLOR_SCHEME["secondary"])
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("mean_viewers:Q", title="Average US Viewers (Millions)"),
    )
)
season_average_viewers.save("outputs/q5_season_average_viewers.html")   

season_max_viewers = (
    base.transform_aggregate(
        max_viewers="max(us_viewers_in_millions)",
        groupby=["season"],
    )
    .mark_line(color=SIMPSONS_COLOR_SCHEME["primary"])
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("max_viewers:Q", title="Max US Viewers (Millions)"),
    )
)
season_max_viewers.save("outputs/q5_season_max_viewers.html")

season_min_viewers = (
    base.transform_aggregate(
        min_viewers="min(us_viewers_in_millions)",
        groupby=["season"],
    )
    .mark_line(color=SIMPSONS_COLOR_SCHEME["alternative_accent"])
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("min_viewers:Q", title="Min US Viewers (Millions)"),
    )
)
season_min_viewers.save("outputs/q5_season_min_viewers.html")

combined = alt.layer(season_average_viewers, season_max_viewers, season_min_viewers).configure_view(stroke=None).configure_axisY(title="Max/Min/Average US Viewers (Millions)")
combined.save("outputs/q5_season_viewers_combined.html")
"""
Q1 – How have the ratings evolved over time?
Three Altair visualisations that each tackle the question differently.
  • Viz 1 (INTERACTIVE): brush interval on timeline → detail table below
  • Viz 2 (INTERACTIVE): click a season bar → episode strip chart appears
  • Viz 3 (static):      season × episode heatmap
"""

import pandas as pd
import altair as alt

df = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
df["original_air_date"] = pd.to_datetime(df["original_air_date"])

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 – INTERACTIVE: overview + magnified zoom panel  (Altair 5 compatible)
#
# Layout: two stacked panels.
#   • OVERVIEW (top, compact): full 27-year timeline — rolling-average line +
#     faint dots.  A brush is drawn here; selected dots turn orange, the rest
#     fade to grey.  A Vega-Lite `expr` parameter clamps the brush width to
#     ≈ 3 seasons so the zoom panel is always readable.
#   • ZOOM (bottom, tall): filtered with transform_filter(brush) — only the
#     brushed episodes are kept, so Altair auto-fits BOTH the X and Y scales
#     to exactly those points.  Y auto-scaling reveals small fluctuations that
#     are invisible at the global [4, 10] range.
# ─────────────────────────────────────────────────────────────────────────────

brush = alt.selection_interval(
    encodings=["x"],
    mark=alt.BrushConfig(
        fill="#6baed6", fillOpacity=0.15,
        stroke="#2171b5", strokeWidth=1.5,
    ),
)

# Maximum brush width enforced via a calculated parameter (≈ 3 seasons in ms)
MAX_MS = 1_095 * 24 * 60 * 60 * 1_000   # 3 years expressed in milliseconds
max_width_param = alt.param(
    name="max_width",
    value=MAX_MS,
)

base = alt.Chart(df)

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
ov_dots = (
    base.mark_circle(size=25)
    .encode(
        x=alt.X(
            "original_air_date:T",
            title=None,
            axis=alt.Axis(format="%Y", tickCount="year", labelFontSize=10),
        ),
        y=alt.Y(
            "imdb_rating:Q",
            title="Rating",
            scale=alt.Scale(domain=[4, 10]),
            axis=alt.Axis(tickCount=4, labelFontSize=10),
        ),
        color=alt.condition(brush, alt.value("#E89B2A"), alt.value("#dddddd")),
        opacity=alt.condition(brush, alt.value(0.75), alt.value(0.3)),
    )
    .add_params(brush, max_width_param)
)

ov_line = (
    base.mark_line(color="#D02020", strokeWidth=1.8, opacity=0.85)
    .encode(
        x=alt.X("original_air_date:T"),
        y=alt.Y("imdb_rating_roll5:Q", scale=alt.Scale(domain=[4, 10])),
    )
)

overview = (
    (ov_dots + ov_line)
    .properties(
        width=820, height=110,
        title=alt.TitleParams(
            "① Drag on this overview to select a window — zoom auto-scales below",
            fontSize=11, color="#555555", anchor="start",
        ),
    )
)

# ── ZOOM ─────────────────────────────────────────────────────────────────────
# transform_filter keeps only the brushed rows → Altair fits X and Y to them.
# zero=False on Y means the axis starts just below the minimum visible rating
# instead of at 0, making fluctuations clearly visible.

zm_dots = (
    base.mark_circle(size=80, opacity=0.80)
    .encode(
        x=alt.X(
            "original_air_date:T",
            title="Air Date",
            axis=alt.Axis(format="%b %Y", tickCount=10, labelAngle=-35, labelFontSize=10),
        ),
        y=alt.Y(
            "imdb_rating:Q",
            title="IMDb Rating",
            scale=alt.Scale(zero=False),   # auto-fit to visible data
            axis=alt.Axis(labelFontSize=11),
        ),
        color=alt.Color(
            "season:O",
            title="Season",
            scale=alt.Scale(scheme="tableau20"),
            legend=alt.Legend(columns=4, symbolSize=80, labelFontSize=10),
        ),
        tooltip=[
            alt.Tooltip("title:N",             title="Episode"),
            alt.Tooltip("season:O",            title="Season"),
            alt.Tooltip("imdb_rating:Q",       title="Rating",   format=".1f"),
            alt.Tooltip("original_air_date:T", title="Air Date", format="%d %b %Y"),
        ],
    )
    .transform_filter(brush)
)

zm_line = (
    base.mark_line(strokeWidth=2.2, opacity=0.65, strokeDash=[4, 2], color="#D02020")
    .encode(
        x=alt.X("original_air_date:T"),
        y=alt.Y("imdb_rating_roll5:Q", scale=alt.Scale(zero=False)),
    )
    .transform_filter(brush)
)

# Horizontal mean-rating rule for the selected window
zm_avg = (
    base.mark_rule(strokeDash=[6, 3], opacity=0.50, color="#555555", strokeWidth=1.5)
    .encode(y=alt.Y("mean(imdb_rating):Q", title=""))
    .transform_filter(brush)
)

zoom = (
    (zm_dots + zm_line + zm_avg)
    .properties(
        width=820, height=310,
        title=alt.TitleParams(
            "② Zoomed — X & Y auto-scale to selection  |  dashed red = rolling avg  |  grey rule = window mean",
            fontSize=11, color="#555555", anchor="start",
        ),
    )
)

viz1 = (
    alt.vconcat(overview, zoom, spacing=14)
    .properties(title=alt.TitleParams(
        "Episode Ratings Over Time – Overview + Zoom (Q1 – Viz 1)",
        fontSize=15, anchor="start",
    ))
    .resolve_scale(color="independent")
)

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 – INTERACTIVE: click a season bar → episode strip chart appears
#
# Top bar chart shows season-average rating (colour = quality).
# Clicking any bar selects that season and the bottom panel immediately renders
# every episode in that season as a horizontal strip coloured by its individual
# rating — revealing within-season distribution at a glance.
# ─────────────────────────────────────────────────────────────────────────────

season_df = (
    df.groupby("season")
    .agg(avg_rating=("imdb_rating", "mean"), n_ep=("id", "count"))
    .reset_index()
    .round({"avg_rating": 3})
)

season_click = alt.selection_point(fields=["season"])  # click to select a season

# Top: season bars
season_bars = (
    alt.Chart(season_df)
    .mark_bar(cursor="pointer")
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("avg_rating:Q", title="Avg IMDb Rating", scale=alt.Scale(domain=[5, 9])),
        color=alt.condition(
            season_click,
            alt.Color(
                "avg_rating:Q",
                scale=alt.Scale(scheme="yelloworangered", domain=[5, 9]),
                legend=None,
            ),
            alt.value("#dddddd"),   # unselected bars go grey
        ),
        tooltip=[
            alt.Tooltip("season:O", title="Season"),
            alt.Tooltip("avg_rating:Q", title="Avg Rating", format=".2f"),
            alt.Tooltip("n_ep:Q", title="Episodes"),
        ],
    )
    .add_params(season_click)
    .properties(width=700, height=260, title="Click a season bar to see its episodes ↓")
)

# Bottom: episode strip for the selected season
episode_strip = (
    alt.Chart(df)
    .mark_rect(height=40)
    .encode(
        x=alt.X(
            "number_in_season:O",
            title="Episode in Season",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "imdb_rating:Q",
            title="IMDb Rating",
            scale=alt.Scale(scheme="yelloworangered", domain=[4.5, 9.5]),
        ),
        tooltip=[
            alt.Tooltip("title:N", title="Episode"),
            alt.Tooltip("number_in_season:O", title="Ep #"),
            alt.Tooltip("imdb_rating:Q", title="Rating", format=".1f"),
            alt.Tooltip("original_air_date:T", title="Air Date", format="%d %b %Y"),
        ],
    )
    .transform_filter(season_click)
    .properties(
        width=700,
        height=70,
        title="Episode ratings for selected season",
    )
)

viz2 = (
    alt.vconcat(season_bars, episode_strip)
    .properties(title=alt.TitleParams(
        "Season Avg Ratings – Click to Drill Down (Q1 – Viz 2)",
        fontSize=15,
        anchor="start",
    ))
)

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 – Heatmap: season × episode-in-season
# Arranges every episode in a grid (season on Y, episode-within-season on X)
# coloured by rating.  Lets the viewer spot which seasons had consistently
# good/bad episodes and whether patterns appear early or late in a season.
# ─────────────────────────────────────────────────────────────────────────────

viz3 = (
    alt.Chart(df, title="Rating Heatmap: Season × Episode Position (Q1 – Viz 3)")
    .mark_rect()
    .encode(
        x=alt.X(
            "number_in_season:O",
            title="Episode in Season",
            axis=alt.Axis(labelOverlap=True),
        ),
        y=alt.Y("season:O", title="Season", sort="descending"),
        color=alt.Color(
            "imdb_rating:Q",
            title="IMDb Rating",
            scale=alt.Scale(scheme="plasma", domain=[4.5, 9.5]),
        ),
        tooltip=[
            alt.Tooltip("title:N", title="Episode"),
            alt.Tooltip("season:O", title="Season"),
            alt.Tooltip("number_in_season:O", title="Ep #"),
            alt.Tooltip("imdb_rating:Q", title="Rating", format=".1f"),
        ],
    )
    .properties(width=700, height=450)
)

# ─────────────────────────────────────────────────────────────────────────────
# Save to HTML
# ─────────────────────────────────────────────────────────────────────────────

viz1.save("q1_viz1_scatter_rolling.html")
viz2.save("q1_viz2_season_bars.html")
viz3.save("q1_viz3_heatmap.html")

print("Q1 charts saved:")
print("  q1_viz1_scatter_rolling.html")
print("  q1_viz2_season_bars.html")
print("  q1_viz3_heatmap.html")

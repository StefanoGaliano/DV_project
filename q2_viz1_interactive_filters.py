"""
q2_viz1_interactive_filters.py  (v4)
──────────────────────────────────────
Fixes vs v3:
  • Multi-select seasons: toggle="true" (string) correctly maps to Vega-Lite
    spec so every click adds/removes independently — no replacement.
  • Zoom/magnify: overview+detail pattern. A small avg-viewers bar chart
    sits below the pills with an interval brush on the time axis. Dragging
    the brush on the overview zooms the main chart x-axis to that window.
    The season pills filter WHICH episodes are shown; the brush controls
    HOW FAR IN you are zoomed. Both work independently and together.
"""

import os
import pandas as pd
import altair as alt

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────

episodes = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
episodes["original_air_date"] = pd.to_datetime(episodes["original_air_date"])

chars = pd.read_csv("data/outputs/simpsons_characters_clean.csv")
locs  = pd.read_csv("data/outputs/simpsons_locations_clean.csv")
lines = pd.read_csv("data/outputs/simpsons_script_lines_clean.csv", low_memory=False)

lines["episode_id"]   = pd.to_numeric(lines["episode_id"],   errors="coerce")
lines["character_id"] = pd.to_numeric(lines["character_id"], errors="coerce")
lines["location_id"]  = pd.to_numeric(lines["location_id"],  errors="coerce")
lines["word_count"]   = pd.to_numeric(lines["word_count"],   errors="coerce")

def to_bool(v):
    if pd.isna(v):          return False
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true", "1", "yes")

lines["speaking_line"] = lines["speaking_line"].apply(to_bool)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  AGGREGATE SCRIPT-LINES → one row per episode
# ─────────────────────────────────────────────────────────────────────────────

ep_agg = (
    lines.groupby("episode_id")
    .agg(
        n_lines          = ("id",            "count"),
        n_speaking_lines = ("speaking_line", "sum"),
        avg_word_count   = ("word_count",    "mean"),
        n_unique_chars   = ("character_id",  "nunique"),
        n_unique_locs    = ("location_id",   "nunique"),
    )
    .reset_index()
)
ep_agg["dialogue_ratio"] = (
    ep_agg["n_speaking_lines"] / ep_agg["n_lines"].replace(0, pd.NA)
).round(3)
ep_agg["avg_word_count"] = ep_agg["avg_word_count"].round(2)

speaking = lines[lines["speaking_line"] & lines["character_id"].notna()]
top_char_id = (
    speaking.groupby(["episode_id", "character_id"])
    .size().reset_index(name="c")
    .sort_values("c", ascending=False)
    .drop_duplicates("episode_id")[["episode_id", "character_id"]]
)
top_char_id = top_char_id.merge(
    chars[["id", "name"]].rename(columns={"id": "character_id", "name": "top_character"}),
    on="character_id", how="left"
).drop(columns="character_id")

with_loc = lines[lines["location_id"].notna()]
top_loc_id = (
    with_loc.groupby(["episode_id", "location_id"])
    .size().reset_index(name="c")
    .sort_values("c", ascending=False)
    .drop_duplicates("episode_id")[["episode_id", "location_id"]]
)
top_loc_id = top_loc_id.merge(
    locs[["id", "name"]].rename(columns={"id": "location_id", "name": "top_location"}),
    on="location_id", how="left"
).drop(columns="location_id")

ep_agg = ep_agg.merge(top_char_id, on="episode_id", how="left")
ep_agg = ep_agg.merge(top_loc_id,  on="episode_id", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  JOIN + ERA
# ─────────────────────────────────────────────────────────────────────────────

df = episodes.merge(
    ep_agg.rename(columns={"episode_id": "id"}),
    on="id", how="left"
)

def era_label(s):
    if s <= 9:    return "Golden Era (S1-9)"
    elif s <= 18: return "Middle Era (S10-18)"
    else:         return "Modern Era (S19-28)"

df["era"] = df["season"].apply(era_label)
era_order  = ["Golden Era (S1-9)", "Middle Era (S10-18)", "Modern Era (S19-28)"]
era_colors = ["#E8A020", "#1A6FA8", "#C03020"]

df_valid = df.dropna(subset=["us_viewers_in_millions"]).copy()

for col in ["n_unique_chars", "n_unique_locs"]:
    df_valid[col] = df_valid[col].fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATE BOUNDS  (tight — no empty space)
# ─────────────────────────────────────────────────────────────────────────────

DATE_MIN = df_valid["original_air_date"].min() - pd.Timedelta(days=20)
DATE_MAX = df_valid["original_air_date"].max() + pd.Timedelta(days=20)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STATIC HELPER FRAMES
# ─────────────────────────────────────────────────────────────────────────────

era_ranges = pd.DataFrame([
    {"era": "Golden Era (S1-9)",   "start": "1989-12-01", "end": "1998-05-31"},
    {"era": "Middle Era (S10-18)", "start": "1998-09-01", "end": "2007-05-31"},
    {"era": "Modern Era (S19-28)", "start": "2007-09-01", "end": "2016-11-01"},
])
era_ranges["start"] = pd.to_datetime(era_ranges["start"])
era_ranges["end"]   = pd.to_datetime(era_ranges["end"])

era_labels_df = pd.DataFrame([
    {"era": "Golden Era (S1-9)",   "date": "1994-01-01", "y": 33},
    {"era": "Middle Era (S10-18)", "date": "2003-01-01", "y": 33},
    {"era": "Modern Era (S19-28)", "date": "2012-06-01", "y": 33},
])
era_labels_df["date"] = pd.to_datetime(era_labels_df["date"])

milestones = (
    df_valid.nlargest(5, "us_viewers_in_millions")
    [["title", "original_air_date", "us_viewers_in_millions",
      "imdb_rating", "season", "top_character", "top_location",
      "n_unique_chars", "n_unique_locs"]]
    .copy()
)
milestones["label"] = milestones["title"].str[:22] + "..."

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

max_chars = int(df_valid["n_unique_chars"].max())
max_locs  = int(df_valid["n_unique_locs"].max())

slider_chars = alt.param(
    name="min_chars",
    value=0,
    bind=alt.binding_range(
        min=0, max=max_chars, step=1,
        name="Min unique characters per episode: ",
    ),
)

slider_locs = alt.param(
    name="min_locs",
    value=0,
    bind=alt.binding_range(
        min=0, max=max_locs, step=1,
        name="Min unique locations per episode:  ",
    ),
)

# ── multi-select season pills ─────────────────────────────────────────────────
# toggle="true" (string) is the correct Vega-Lite value that means
# "add/remove on every click independently" without needing shift.
# empty=True means: nothing selected → show all.
season_sel = alt.selection_point(
    name="season_sel",
    fields=["season"],
    toggle="true",
    empty=True,
)

# ── brush on the overview strip → zooms the main chart x-axis ────────────────
# encodings="x" restricts the brush to the time axis only.
# empty selection on the brush = show the full range (handled via scale domain).
brush = alt.selection_interval(
    name="brush",
    encodings=["x"],
    empty=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FILTER EXPRESSION
# ─────────────────────────────────────────────────────────────────────────────

filter_expr = (
    "(datum.n_unique_chars >= min_chars) && "
    "(datum.n_unique_locs  >= min_locs)"
)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  BUILD MAIN CHART LAYERS
#     X-axis domain is driven by the brush selection via a shared param.
#     When the brush is empty, the domain falls back to the full range.
# ─────────────────────────────────────────────────────────────────────────────

# Era background bands
bands = (
    alt.Chart(era_ranges)
    .mark_rect(opacity=0.08)
    .encode(
        x=alt.X("start:T",
                scale=alt.Scale(domain=brush)),
        x2="end:T",
        color=alt.Color(
            "era:N",
            scale=alt.Scale(domain=era_order, range=era_colors),
            legend=None,
        ),
    )
)

era_text = (
    alt.Chart(era_labels_df)
    .mark_text(fontSize=10, opacity=0.45, fontStyle="italic")
    .encode(
        x=alt.X("date:T",
                scale=alt.Scale(domain=brush)),
        y=alt.Y("y:Q"),
        text="era:N",
        color=alt.Color(
            "era:N",
            scale=alt.Scale(domain=era_order, range=era_colors),
            legend=None,
        ),
    )
)

# Area — filtered by season selection
area = (
    alt.Chart(df_valid)
    .mark_area(
        line={"color": "#4488BB", "strokeWidth": 1.2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#1A6FA8", offset=0),
                alt.GradientStop(color="white",   offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.25,
    )
    .encode(
        x=alt.X("original_air_date:T",
                title="Air Date",
                axis=alt.Axis(format="%b %Y", tickCount=8),
                scale=alt.Scale(domain=brush)),
        y=alt.Y("us_viewers_in_millions:Q",
                title="US Viewers (millions)",
                scale=alt.Scale(zero=False)),
    )
    .transform_filter(season_sel)
)

# Dots — filtered by sliders + season
dots = (
    alt.Chart(df_valid)
    .mark_circle(size=65)
    .encode(
        x=alt.X("original_air_date:T",
                scale=alt.Scale(domain=brush)),
        y=alt.Y("us_viewers_in_millions:Q"),
        color=alt.Color(
            "imdb_rating:Q",
            title="IMDB Rating",
            scale=alt.Scale(scheme="redyellowgreen", domain=[5, 9.5]),
            legend=alt.Legend(orient="right", title="IMDB Rating"),
        ),
        opacity=alt.condition(
            alt.datum.dialogue_ratio > 0,
            alt.Opacity("dialogue_ratio:Q",
                        scale=alt.Scale(range=[0.35, 0.95]),
                        legend=None),
            alt.value(0.45),
        ),
        tooltip=[
            alt.Tooltip("title:N",                  title="Episode"),
            alt.Tooltip("season:O",                 title="Season"),
            alt.Tooltip("original_air_date:T",      title="Air Date",         format="%d %b %Y"),
            alt.Tooltip("era:N",                    title="Era"),
            alt.Tooltip("us_viewers_in_millions:Q", title="Viewers (M)",      format=".2f"),
            alt.Tooltip("imdb_rating:Q",            title="IMDB Rating"),
            alt.Tooltip("n_lines:Q",                title="Total Lines"),
            alt.Tooltip("n_speaking_lines:Q",       title="Speaking Lines"),
            alt.Tooltip("dialogue_ratio:Q",         title="Dialogue Ratio",   format=".0%"),
            alt.Tooltip("avg_word_count:Q",         title="Avg Words / Line", format=".1f"),
            alt.Tooltip("n_unique_chars:Q",         title="Unique Characters"),
            alt.Tooltip("n_unique_locs:Q",          title="Unique Locations"),
            alt.Tooltip("top_character:N",          title="Top Character"),
            alt.Tooltip("top_location:N",           title="Top Location"),
        ],
    )
    .transform_filter(filter_expr)
    .transform_filter(season_sel)
    .add_params(slider_chars, slider_locs)
)

# Milestone triangles
milestone_pts = (
    alt.Chart(milestones)
    .mark_point(shape="triangle-up", size=220, filled=True,
                color="#C03020", opacity=0.9)
    .encode(
        x=alt.X("original_air_date:T",
                scale=alt.Scale(domain=brush)),
        y="us_viewers_in_millions:Q",
        tooltip=[
            alt.Tooltip("title:N",                  title="Top Episode"),
            alt.Tooltip("us_viewers_in_millions:Q", title="Viewers (M)",  format=".1f"),
            alt.Tooltip("imdb_rating:Q",            title="IMDB Rating"),
            alt.Tooltip("season:O",                 title="Season"),
            alt.Tooltip("top_character:N",          title="Top Character"),
            alt.Tooltip("top_location:N",           title="Top Location"),
            alt.Tooltip("n_unique_chars:Q",         title="Unique Characters"),
            alt.Tooltip("n_unique_locs:Q",          title="Unique Locations"),
        ],
    )
    .transform_filter(season_sel)
)

milestone_labels = (
    alt.Chart(milestones)
    .mark_text(align="left", dx=7, dy=-5, fontSize=9,
               color="#C03020", fontWeight="bold")
    .encode(
        x=alt.X("original_air_date:T",
                scale=alt.Scale(domain=brush)),
        y="us_viewers_in_millions:Q",
        text="label:N",
    )
    .transform_filter(season_sel)
)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SEASON PILL STRIP  (multi-select)
# ─────────────────────────────────────────────────────────────────────────────

season_summary = (
    df_valid.groupby(["season", "era"])
    .agg(avg_viewers=("us_viewers_in_millions", "mean"))
    .reset_index()
)

season_pills = (
    alt.Chart(season_summary)
    .mark_rect(cornerRadius=4, height=22)
    .encode(
        x=alt.X(
            "season:O",
            title="Click seasons to filter (multi-select, click again to deselect)",
            axis=alt.Axis(labelAngle=0, titleFontSize=10, titleColor="#555"),
        ),
        color=alt.condition(
            season_sel,
            alt.Color("era:N",
                      scale=alt.Scale(domain=era_order, range=era_colors),
                      legend=None),
            alt.value("#ddd"),
        ),
        tooltip=[
            alt.Tooltip("season:O",      title="Season"),
            alt.Tooltip("era:N",         title="Era"),
            alt.Tooltip("avg_viewers:Q", title="Avg Viewers (M)", format=".1f"),
        ],
    )
    .add_params(season_sel)
    .properties(width=860, height=34)
)

season_labels = (
    alt.Chart(season_summary)
    .mark_text(fontSize=9, fontWeight="bold")
    .encode(
        x=alt.X("season:O"),
        text=alt.Text("season:O"),
        color=alt.condition(season_sel, alt.value("white"), alt.value("#999")),
    )
)

pills_chart = season_pills + season_labels

# ─────────────────────────────────────────────────────────────────────────────
# 10. OVERVIEW STRIP  (drag to zoom main chart x-axis)
#     Shows avg viewers per episode as a thin area chart across the full
#     timeline. Dragging a brush window on it zooms the main chart x-axis.
#     Filtered by season_sel so it matches what is shown above.
# ─────────────────────────────────────────────────────────────────────────────

overview_area = (
    alt.Chart(df_valid)
    .mark_area(color="#1A6FA8", opacity=0.35,
               line={"color": "#1A6FA8", "strokeWidth": 1})
    .encode(
        x=alt.X("original_air_date:T",
                axis=alt.Axis(format="%Y", title=None),
                scale=alt.Scale(
                    domain=[DATE_MIN.isoformat(), DATE_MAX.isoformat()])),
        y=alt.Y("us_viewers_in_millions:Q",
                axis=None,
                scale=alt.Scale(zero=False)),
    )
    .transform_filter(season_sel)
)

overview_brush = (
    alt.Chart(df_valid)
    .mark_point(size=0)           # invisible marks — just to host the brush
    .encode(
        x=alt.X("original_air_date:T",
                scale=alt.Scale(
                    domain=[DATE_MIN.isoformat(), DATE_MAX.isoformat()])),
        y=alt.Y("us_viewers_in_millions:Q"),
    )
    .add_params(brush)
    .properties(width=860, height=50,
                title=alt.TitleParams(
                    "Drag to zoom the chart above   (drag on empty area to reset)",
                    fontSize=9, color="#888", anchor="start"))
)

overview = overview_area + overview_brush

# ─────────────────────────────────────────────────────────────────────────────
# 11. ASSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

main_chart = (
    (bands + era_text + area + dots + milestone_pts + milestone_labels)
    .properties(
        width=860, height=380,
        title=alt.TitleParams(
            "US Viewership Over Time",
            subtitle=[
                "Dot colour = IMDB rating (red to green)  |  Dot opacity = dialogue ratio",
                "Sliders filter by min cast/location breadth  |  "
                "Season pills filter episodes  |  Drag overview to zoom  |  "
                "Triangles = Top-5 most-watched",
            ],
            fontSize=16,
            subtitleFontSize=10,
            subtitleColor="#666",
        ),
    )
)

viz1 = alt.vconcat(
    main_chart,
    pills_chart,
    overview,
    spacing=8,
).configure_view(strokeWidth=0)

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────────────────────────────────────

out = "outputs/q2_viz1_interactive_filters.html"
viz1.save(out)
print(f"Viz 1 saved -> {out}")

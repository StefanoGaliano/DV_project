import os, warnings
import pandas as pd
import numpy as np
import altair as alt

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ─── colour palette (Okabe-Ito, colour-blind safe) ────────────────────────────
C_GOLDEN  = "#E69F00"
C_MIDDLE  = "#0072B2"
C_MODERN  = "#D55E00"
C_ACCENT  = "#009E73"
C_NEUTRAL = "#999999"

ERA_ORDER  = ["Golden Era (S1-9)", "Middle Era (S10-18)", "Modern Era (S19-28)"]
ERA_COLORS = [C_GOLDEN, C_MIDDLE, C_MODERN]

def era_label(s):
    if s <= 9:    return "Golden Era (S1-9)"
    elif s <= 18: return "Middle Era (S10-18)"
    else:         return "Modern Era (S19-28)"

ERA_SCALE = alt.Scale(domain=ERA_ORDER, range=ERA_COLORS)

def era_color(legend=True):
    return alt.Color(
        "era:N", title="Era", scale=ERA_SCALE,
        legend=alt.Legend(orient="top") if legend else None,
    )

TITLE_FONT = 14
AXIS_FONT  = 11

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────

eps   = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")
eps["original_air_date"] = pd.to_datetime(eps["original_air_date"])
eps["era"] = eps["season"].apply(era_label)

chars = pd.read_csv("data/outputs/simpsons_characters_clean.csv")
locs  = pd.read_csv("data/outputs/simpsons_locations_clean.csv")

LINES_PATH = "data/outputs/simpsons_script_lines_clean.csv"
has_lines  = os.path.exists(LINES_PATH)

if has_lines:
    lines = pd.read_csv(LINES_PATH, low_memory=False)
    for col in ["episode_id","character_id","location_id","word_count"]:
        lines[col] = pd.to_numeric(lines[col], errors="coerce")
    def to_bool(v):
        if pd.isna(v): return False
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("true","1","yes")
    lines["speaking_line"] = lines["speaking_line"].apply(to_bool)

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

    speaking = lines[lines["speaking_line"] & lines["character_id"].notna()]
    top_char = (
        speaking.groupby(["episode_id","character_id"]).size()
        .reset_index(name="c").sort_values("c", ascending=False)
        .drop_duplicates("episode_id")[["episode_id","character_id"]]
    )
    top_char = top_char.merge(
        chars[["id","name"]].rename(columns={"id":"character_id","name":"top_character"}),
        on="character_id", how="left").drop(columns="character_id")

    top_loc = (
        lines[lines["location_id"].notna()]
        .groupby(["episode_id","location_id"]).size()
        .reset_index(name="c").sort_values("c", ascending=False)
        .drop_duplicates("episode_id")[["episode_id","location_id"]]
    )
    top_loc = top_loc.merge(
        locs[["id","name"]].rename(columns={"id":"location_id","name":"top_location"}),
        on="location_id", how="left").drop(columns="location_id")

    gender_map = chars[["id","gender"]].rename(columns={"id":"character_id"})
    speaking_g = speaking.merge(gender_map, on="character_id", how="left")
    gender_ep = (
        speaking_g.groupby("episode_id")
        .agg(
            n_female   = ("gender", lambda x: (x == "female").sum()),
            n_gendered = ("gender", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    gender_ep["pct_female"] = (
        gender_ep["n_female"] / gender_ep["n_gendered"].replace(0, pd.NA)
    ).round(3)

    ep_agg = (ep_agg
              .merge(top_char, on="episode_id", how="left")
              .merge(top_loc,  on="episode_id", how="left")
              .merge(gender_ep[["episode_id","pct_female"]], on="episode_id", how="left"))

    eps = eps.merge(ep_agg.rename(columns={"episode_id":"id"}), on="id", how="left")
    for col in ["n_unique_chars","n_unique_locs"]:
        eps[col] = eps[col].fillna(0).astype(int)

    season_lines = (
        eps.groupby("season")
        .agg(
            avg_chars  = ("n_unique_chars", "mean"),
            avg_locs   = ("n_unique_locs",  "mean"),
            avg_dlg    = ("dialogue_ratio",  "mean"),
            avg_words  = ("avg_word_count",  "mean"),
            avg_female = ("pct_female",      "mean"),
        )
        .reset_index()
    )
else:
    season_lines = None

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEASON AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

season = (
    eps.groupby(["season","era"])
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
season["ci_viewers"] = (season["std_viewers"] / np.sqrt(season["n_episodes"]) * 1.96).round(3)
# Label strings baked in Python — no tooltip needed
season["avg_rating_lbl"]  = season["avg_rating"].round(1).astype(str)
season["avg_viewers_lbl"] = season["avg_viewers"].round(1).astype(str)

if season_lines is not None:
    season = season.merge(season_lines, on="season", how="left")
    season["avg_chars_lbl"] = season["avg_chars"].round(1).astype(str)
    season["avg_locs_lbl"]  = season["avg_locs"].round(1).astype(str)

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




# --- Configuration Constants (Assumed from your setup) ---
C_NEUTRAL = "#888888"
ERA_COLORS = ["#E69F00", "#56B4E9", "#009E73"] # Colorblind-safe (Vermillion, Sky Blue, Bluish Green)
AXIS_FONT = 11
TITLE_FONT = 16

def era_color(legend=True):
    return alt.Color("era_name:N",
                     scale=alt.Scale(
                         domain=["Golden Era", "Middle Era", "Modern Era"],
                         range=ERA_COLORS),
                     legend=alt.Legend(orient="top", title="Show Era") if legend else None)

import os, warnings
import pandas as pd
import numpy as np
import altair as alt

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ─── colour palette (Okabe-Ito, colour-blind safe) ────────────────────────────
C_GOLDEN  = "#E69F00"
C_MIDDLE  = "#0072B2"
C_MODERN  = "#D55E00"
C_ACCENT  = "#009E73"
C_NEUTRAL = "#999999"

ERA_ORDER  = ["Golden Era (S1-9)", "Middle Era (S10-18)", "Modern Era (S19-28)"]
ERA_COLORS = [C_GOLDEN, C_MIDDLE, C_MODERN]

def era_label(s):
    if s <= 9:    return "Golden Era (S1-9)"
    elif s <= 18: return "Middle Era (S10-18)"
    else:          return "Modern Era (S19-28)"

ERA_SCALE = alt.Scale(domain=ERA_ORDER, range=ERA_COLORS)

def era_color(legend=True):
    return alt.Color(
        "era:N", title="Era", scale=ERA_SCALE,
        legend=alt.Legend(orient="top", padding=10) if legend else None,
    )

TITLE_FONT = 14
AXIS_FONT  = 11

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & AGGREGATE (Simplified for context)
# ─────────────────────────────────────────────────────────────────────────────
# [Assuming 'season' dataframe is already built from your provided loading logic]

# Ensure numeric formatting for labels
season["avg_viewers_lbl"] = season["avg_viewers"].round(1).astype(str)

# Calculate scaling for the location bars to keep them at the bottom (0-4 range)
max_locs = season["avg_locs"].max() if "avg_locs" in season.columns else 1

# ─── 2. CI BAND (Viewership) ────────────────────────────────────────────────
band_v = (
    alt.Chart(season.dropna(subset=["avg_viewers", "ci_viewers"]))
    .mark_area(opacity=0.15)
    .encode(
        x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0, titleFontSize=AXIS_FONT)),
        y=alt.Y("min_v:Q", title="US Viewers (millions)", scale=alt.Scale(domain=[0, 35])),
        y2="max_v:Q",
        color=era_color(legend=False),
    )
    .transform_calculate(
        min_v="datum.avg_viewers - datum.ci_viewers",
        max_v="datum.avg_viewers + datum.ci_viewers",
    )
)

locs_bar = (
    alt.Chart(season)
    .mark_bar(opacity=0.2)
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("scaled_locs:Q"),
        tooltip=[alt.Tooltip("avg_locs:Q", title="Avg Unique Locations")],
        # Use alt.value for the color, and alt.Color with a title for the legend
        color=alt.value("#CCCCCC"), 
        opacity=alt.Opacity("legend_label:N", 
            scale=alt.Scale(range=[0.2]), 
            legend=alt.Legend(title="Contextual Info", orient="right")
        )
    )
    .transform_calculate(
        scaled_locs=f"(datum.avg_locs / {max_locs}) * 4",
        legend_label="'Avg Unique Locations'" # Create a string for the legend
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
        color=era_color(legend=False),
    )
)

# ─── 5. SEASON AVERAGE LINE + POINTS ────────────────────────────────────────
line_v = (
    alt.Chart(season.dropna(subset=["avg_viewers"]))
    .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=60, filled=True))
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_viewers:Q"),
        color=era_color(),
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

# ─── 7. ERA ANNOTATIONS (Floating text) ─────────────────────────────────────
era_annot_df = pd.DataFrame([
    {"season": 5,  "y": 32, "era": "Golden Era (S1-9)"},
    {"season": 14, "y": 32, "era": "Middle Era (S10-18)"},
    {"season": 23, "y": 32, "era": "Modern Era (S19-28)"},
])
era_annot_v = (
    alt.Chart(era_annot_df)
    .mark_text(fontSize=10, fontStyle="italic", opacity=0.6, fontWeight="bold")
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("y:Q"),
        text="era:N",
        color=alt.Color("era:N", scale=ERA_SCALE, legend=None),
    )
)

# ─── 8. COMBINED FINAL CHART ────────────────────────────────────────────────
chart_q2 = (
    # Layer order: Bars at the back, Labels/Annotations at the front
    alt.layer(locs_bar, band_v, range_v, line_v, lbl_v, era_annot_v)
    .properties(
        width=800, 
        height=350,
        title=alt.TitleParams(
            "Q2 — Evolution of US Viewership and Location Variety",
            subtitle=[
                "Numbers above dots = Avg Viewers (Millions) · Vertical Rules = Min-Max Range per Season",
                "Faint Grey Bars = Relative count of unique locations per script (normalized to bottom)"
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

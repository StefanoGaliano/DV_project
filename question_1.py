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
# --- Setup and Data Mockup (as per your script logic) ---
# Assuming 'season' and 'eps_r' and 'top5_r' are already defined in your environment
# Here is the revised Chart Q1 logic:

# --- Color Palette and Scales ---
C_GOLDEN  = "#E69F00"
C_MIDDLE  = "#0072B2"
C_MODERN  = "#D55E00"
C_NEUTRAL = "#999999"

ERA_ORDER  = ["Golden Era (S1-9)", "Middle Era (S10-18)", "Modern Era (S19-28)"]
ERA_COLORS = [C_GOLDEN, C_MIDDLE, C_MODERN]
ERA_SCALE  = alt.Scale(domain=ERA_ORDER, range=ERA_COLORS)

def era_color(legend=True):
    return alt.Color(
        "era:N", title="Era", scale=ERA_SCALE,
        legend=alt.Legend(orient="top") if legend else None,
    )

# --- Chart Q1: Single Y-Axis with Clean Labels ---

# 1. Individual Episode Dots (Faint Background)
dots_r = (
    alt.Chart(eps_r)
    .mark_circle(size=15, opacity=0.12, color=C_NEUTRAL)
    .encode(
        x=alt.X("season:O", title="Season"),
        y=alt.Y("imdb_rating:Q", title="IMDB Rating", scale=alt.Scale(domain=[4, 10])),
    )
)

# 2. 95% Confidence Interval Band
band_r = (
    alt.Chart(season.dropna(subset=["avg_rating", "ci_rating"]))
    .mark_area(opacity=0.15)
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("min_r:Q"),
        y2="max_r:Q",
        color=era_color(legend=False),
    )
    .transform_calculate(
        min_r="datum.avg_rating - datum.ci_rating",
        max_r="datum.avg_rating + datum.ci_rating",
    )
)

# 3. Season Average Line and Fixed-Size Points
line_r = (
    alt.Chart(season.dropna(subset=["avg_rating"]))
    .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=50, filled=True))
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_rating:Q"),
        color=era_color(),
    )
)

# --- 4. DATA LABELS (Numbers above every dot) ---
lbl_dots_r = (
    alt.Chart(season.dropna(subset=["avg_rating"]))
    .mark_text(fontSize=10, fontWeight="bold", dy=-15, color="#444") 
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("avg_rating:Q"),
        text=alt.Text("avg_rating:Q", format=".1f")
    )
)

## --- 5. TOP-5 HIGHLIGHTS (Corrected to force Episode Titles into Legend) ---
top5_r = eps_r.nlargest(5, "imdb_rating").copy()

annot_pt_r = (
    alt.Chart(top5_r)
    .mark_point(shape="diamond", size=150, filled=True, stroke="black", strokeWidth=1)
    .encode(
        x=alt.X("season:O"),
        y=alt.Y("imdb_rating:Q"),
        # We use 'title' here to ensure every episode gets its own name in the legend
        color=alt.Color("title:N", 
            # Using a high-contrast palette for the 5 episodes
            scale=alt.Scale(scheme="category10"), 
            legend=alt.Legend(
                title="Highest Rated Episodes",
                orient="right",
                symbolType="diamond",
                padding=10,
                # This ensures the legend doesn't try to merge with the 'Era' legend
                offset=20 
            )
        )
    )
)

# --- Combined Chart ---
chart_q1_clean = (
    # Ensure annot_pt_r is added at the end
    (dots_r + band_r + line_r + lbl_dots_r + annot_pt_r)
    .properties(
        width=850, 
        height=400,
        title=alt.TitleParams(
            "Q1 — Evolution of IMDB Ratings by Season",
            subtitle=[
                "Numbers above dots represent average season ratings. Diamond markers indicate the top 5 highest-rated episodes.",
                "Line = Season Average · Shaded Area = 95% Confidence Interval"
            ],
            fontSize=16, subtitleFontSize=11, subtitleColor="#444"
        )
    )
    .resolve_scale(
        # THIS IS KEY: It tells Altair that the color for the points 
        # is independent of the color for the lines/eras.
        color='independent' 
    )
    .configure_view(stroke=None)
    .configure_legend(
        cornerRadius=5,
        fillColor='white',
        strokeColor='#DDD',
        padding=10
    )
)

chart_q1_clean.save("outputs/q1_ratings_clean.html")

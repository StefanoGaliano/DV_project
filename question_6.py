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

if season_lines is not None:
    # 0. FILTER DATA: Drop seasons without script data (27 & 28)
    # This ensures the X-axis ends at season 26 where data is actually present
    season_filtered = season[~season["season"].isin([27, 28])].copy()

    # 1. CALCULATE REASONABLE CEILING
    # We want the orange line to stay below the bar labels.
    max_chars = season_filtered["avg_chars"].max()
    max_locs = season_filtered["avg_locs"].max()
    
    # Scaling factor to keep the orange line in the lower 65% of the chart height
    safe_zone_factor = (max_chars * 0.65) / max_locs

    # 2. BARS (Characters)
    bar_chars = (
        alt.Chart(season_filtered)
        .mark_bar(opacity=0.4, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("season:O", title="Season",
                    axis=alt.Axis(labelAngle=0, titleFontSize=AXIS_FONT)),
            y=alt.Y("avg_chars:Q", title="Complexity Metrics (Avg per Episode)",
                    scale=alt.Scale(domain=[0, max_chars + 5])),
            color=alt.Color(
                "avg_rating:Q", title="Avg IMDB",
                scale=alt.Scale(scheme="brownbluegreen", domain=[6.5, 8.5]),
                legend=alt.Legend(orient="right"),
            ),
        )
    )

    # 3. BAR LABELS (Top of bars)
    bar_lbl_q6 = (
        alt.Chart(season_filtered)
        .mark_text(fontSize=8, dy=-8, fontWeight="bold", color="#333")
        .encode(
            x=alt.X("season:O"),
            y=alt.Y("avg_chars:Q"),
            text=alt.Text("avg_chars_lbl:N"),
        )
    )

    # 4. STANDARDIZED ORANGE LINE (Locations)
    line_locs = (
        alt.Chart(season_filtered)
        .mark_line(color="#E67E22", strokeWidth=2.5, 
                   point=alt.OverlayMarkDef(size=30, filled=True, color="#E67E22"))
        .encode(
            x=alt.X("season:O"),
            y=alt.Y("scaled_locs:Q"),
        )
        .transform_calculate(
            scaled_locs=f"datum.avg_locs * {safe_zone_factor}"
        )
    )

    # 5. LINE LABELS (Orange labels - positioned below points)
    locs_lbl = (
        alt.Chart(season_filtered)
        .mark_text(fontSize=8, color="#E67E22", fontWeight="bold", dy=12)
        .encode(
            x=alt.X("season:O"),
            y=alt.Y("scaled_locs:Q"),
            text=alt.Text("avg_locs_lbl:N"),
        )
        .transform_calculate(
            scaled_locs=f"datum.avg_locs * {safe_zone_factor}"
        )
    )

    # 6. FINAL ASSEMBLY
    chart_q6 = (
        alt.layer(bar_chars, bar_lbl_q6, line_locs, locs_lbl)
        .properties(
            width=780, height=320,
            title=alt.TitleParams(
                "Q6 — Production Complexity Trends (Seasons 1-26)",
                subtitle=[
                    "Bars = Avg Characters per Episode · Orange Line = Avg Locations per Episode (Normalized Scale)",
                    "Labels represent actual counts · Seasons 27-28 excluded due to missing script data."
                ],
                fontSize=TITLE_FONT, subtitleFontSize=10, subtitleColor="#444",
            ),
        )
        .configure_view(stroke=None)
    )

    chart_q6.save("outputs/q6_final_complexity.html")

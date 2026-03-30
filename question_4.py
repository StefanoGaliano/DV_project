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



DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday", "Sunday"]

wd_agg = (
    eps_v.groupby("air_dayofweek")
    .agg(
        mean_viewers = ("us_viewers_in_millions", "mean"),
        std_viewers  = ("us_viewers_in_millions", "std"),
        n_episodes   = ("us_viewers_in_millions", "count"),
    )
    .reset_index()
)
wd_agg["ci"] = (wd_agg["std_viewers"] / np.sqrt(wd_agg["n_episodes"]) * 1.96).round(3)
wd_agg["ci_lo"] = wd_agg["mean_viewers"] - wd_agg["ci"]
wd_agg["ci_hi"] = wd_agg["mean_viewers"] + wd_agg["ci"]
wd_agg["reliable"] = wd_agg["n_episodes"] >= 5
# Label: mean + n — shown directly on bars
wd_agg["bar_label"] = (
    wd_agg["mean_viewers"].round(1).astype(str) + "M  (n="
    + wd_agg["n_episodes"].astype(str) + ")"
)
# Unreliable days get an asterisk warning appended
wd_agg.loc[~wd_agg["reliable"], "bar_label"] = (
    wd_agg.loc[~wd_agg["reliable"], "bar_label"] + " *"
)
# 1. Physical Cleanup: Ensure no trailing spaces and enforce Categorical order
# 1. Create a mapping dictionary for absolute control
mapping = {day: i for i, day in enumerate(DAY_ORDER)}

# 2. Clean the data and create a hidden sort column
wd_agg['air_dayofweek'] = wd_agg['air_dayofweek'].astype(str).str.strip()
wd_agg['day_num'] = wd_agg['air_dayofweek'].map(mapping)

# Drop any days that didn't match our DAY_ORDER (optional safety net)
wd_agg = wd_agg.dropna(subset=['day_num']).sort_values("day_num")

# 3. Define the X-axis using the Sort Column
# This forces Altair to use the order of day_num (0, 1, 2...) 
# while displaying the names from air_dayofweek
x_encoding = alt.X(
    "air_dayofweek:N", 
    title="Day of Week",
    sort=DAY_ORDER, # Explicitly sort by the list order
    axis=alt.Axis(labelAngle=0, titleFontSize=AXIS_FONT)
)

# --- 1. Bars ---
bars_wd = (
    alt.Chart(wd_agg)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=0.85)
    .encode(
        x=x_encoding,
        y=alt.Y("mean_viewers:Q", title="Avg US Viewers (millions)", scale=alt.Scale(domain=[0, 32])),
        color=alt.condition(
            alt.datum.reliable,
            alt.value(C_MIDDLE),
            alt.value("#DCDCDC")
        )
    )
)

# --- 2. Error Bars (Shared X) ---
errbar_wd = (
    alt.Chart(wd_agg[wd_agg["reliable"]])
    .mark_errorbar(color="#333", ticks=True, thickness=2, size=12)
    .encode(
        x=x_encoding,
        y=alt.Y("ci_lo:Q"),
        y2="ci_hi:Q"
    )
)

# --- 3. Labels (Shared X) ---
bar_lbl_wd = (
    alt.Chart(wd_agg)
    .mark_text(fontSize=10, fontWeight="bold", dy=-15)
    .encode(
        x=x_encoding,
        y=alt.Y("mean_viewers:Q"),
        text=alt.Text("bar_label:N"),
        color=alt.condition(alt.datum.reliable, alt.value("#222"), alt.value("#888"))
    )
)

# --- 4. Footnote (Using a specific index to prevent alignment drift) ---
footnote_wd = (
    alt.Chart(pd.DataFrame([{"day": "Monday", "v": -3}]))
    .mark_text(fontSize=9, fontStyle="italic", color="#666", align="left")
    .encode(
        x=alt.X("day:N", sort=DAY_ORDER),
        y=alt.Y("v:Q"),
        text=alt.value("* Fewer than 5 episodes: the mean value is statistically unreliable.")
    )
)

# --- 5. Final Assembly ---
chart_q4 = (
    (bars_wd + errbar_wd + bar_lbl_wd + footnote_wd)
    .properties(
        width=600, height=350,
        title=alt.TitleParams(
            "Q4 — Viewership Performance by Broadcast Day",
            subtitle=["Blue = Reliable · Grey = Small Sample · Error bars = 95% CI"],
            fontSize=TITLE_FONT
        )
    )
    .configure_view(stroke=None)
    .configure_axis(grid=False)
)

chart_q4.save("outputs/q4_viewers_by_day.html")

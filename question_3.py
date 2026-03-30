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

scatter = (
    alt.Chart(eps_rv)
    .mark_circle(opacity=0.50)
    .encode(
        x=alt.X("us_viewers_in_millions:Q", title="US Viewers (millions)",
                axis=alt.Axis(titleFontSize=AXIS_FONT)),
        y=alt.Y("imdb_rating:Q", title="IMDB Rating",
                scale=alt.Scale(domain=[4, 9.5])),
        color=era_color(),
        size=alt.Size("imdb_votes:Q", title="IMDB Votes",
                      scale=alt.Scale(range=[12, 180]),
                      legend=alt.Legend(orient="right")),
    )
)

reg_line = (
    alt.Chart(reg_df)
    .mark_line(color=C_ACCENT, strokeWidth=2.2, strokeDash=[6, 3])
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
    alt.Chart(reg_eq_df)
    .mark_text(fontSize=10, color=C_ACCENT, fontStyle="italic", fontWeight="bold")
    .encode(x="x:Q", y="y:Q", text="label:N")
)

# Annotate key outliers directly on the chart
# Top 3 rated, 2 lowest rated, 3 highest-viewer, 1 highest-vote
def get_outliers(df):
    seen = set()
    rows = []
    for subset in [
        df.nlargest(3, "imdb_rating"),
        df.nsmallest(2, "imdb_rating"),
        df.nlargest(3, "us_viewers_in_millions"),
        df.nlargest(1, "imdb_votes"),
    ]:
        for _, r in subset.iterrows():
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return pd.DataFrame(rows)

# --- 1. Identify and Prepare Outliers ---
# --- 1. Identify Outliers & Special Encodings ---
outliers = get_outliers(eps_rv).copy()

# Create a categorical column for Dialogue Ratio (Binary is clearer than opacity)
# High dialogue = Top 25% of episodes
dlg_threshold = eps_rv["dialogue_ratio"].quantile(0.75)
eps_rv["dialogue_type"] = eps_rv["dialogue_ratio"].apply(
    lambda x: "High Dialogue" if x > dlg_threshold else "Standard"
)

# Create a categorical column for Votes (Thick stroke for highly voted)
vote_threshold = eps_rv["imdb_votes"].quantile(0.90)
eps_rv["is_popular"] = eps_rv["imdb_votes"] > vote_threshold

# --- 2. Main Scatter Plot ---
scatter = (
    alt.Chart(eps_rv)
    .mark_point(size=60, filled=True)
    .encode(
        x=alt.X("us_viewers_in_millions:Q", title="US Viewers (millions)",
                axis=alt.Axis(titleFontSize=AXIS_FONT)),
        y=alt.Y("imdb_rating:Q", title="IMDB Rating",
                scale=alt.Scale(domain=[4, 10])),
        color=era_color(),
        # SHAPE represents Dialogue Ratio now
        shape=alt.Shape("dialogue_type:N", title="Dialogue Ratio",
                       scale=alt.Scale(domain=["Standard", "High Dialogue"], 
                                       range=["circle", "square"])),
        # STROKE represents Popularity (Votes)
        stroke=alt.condition(
            alt.datum.is_popular, 
            alt.value("black"), 
            alt.value(None)
        ),
        strokeWidth=alt.condition(alt.datum.is_popular, alt.value(1.5), alt.value(0))
    )
)

# --- 3. Highlighted Outliers (Diamonds in Legend) ---
annot_pts = (
    alt.Chart(outliers)
    .mark_point(shape="diamond", size=160, filled=True, stroke="black", strokeWidth=1.2)
    .encode(
        x=alt.X("us_viewers_in_millions:Q"),
        y=alt.Y("imdb_rating:Q"),
        # Map title to color to force names into a dedicated Legend
        color=alt.Color("title:N", 
            title="Legendary Episodes",
            scale=alt.Scale(scheme="category10"),
            legend=alt.Legend(orient="right", symbolType="diamond", columns=1)
        )
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

reg_annot = (
    alt.Chart(pd.DataFrame([{"x": 22, "y": 4.2, "label": f"r = {r_value:.2f}"}]))
    .mark_text(fontSize=12, color="#444", fontWeight="bold")
    .encode(x="x:Q", y="y:Q", text="label:N")
)

# --- 5. Combined Final Chart ---
chart_q3 = (
    (scatter + reg_line + reg_annot + annot_pts)
    .resolve_scale(color="independent", shape="independent") # Keep legends separate
    .properties(
        width=750, height=450,
        title=alt.TitleParams(
            "Q3 — Correlation: Quality vs. Popularity",
            subtitle=[
                "Circles = Standard · Squares = High Dialogue Ratio (>75th percentile)",
                "Black Border = High Vote Count (>90th percentile) · Diamonds = Key Outliers (see legend)",
                "The dashed line shows the general downward trend as the show aged."
            ],
            fontSize=TITLE_FONT, subtitleFontSize=10, subtitleColor="#444",
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

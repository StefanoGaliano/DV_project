"""
Preprocessing script for the Simpsons dataset.
Produces clean CSVs used by the Streamlit visualization app.

Input files (raw):
    simpsons_script_lines.csv
    simpsons_episodes.csv
    simpsons_characters.csv
    simpsons_locations.csv

Output files (clean, written to ./clean/):
    clean_script_lines.csv      — one row per spoken line, enriched with character/episode info
    clean_episodes.csv          — one row per episode
    clean_characters.csv        — one row per character (main cast only)
    agg_words_by_char.csv       — total words & sentences per character  (Q1, Q5)
    agg_words_by_char_season.csv— words & sentences per character×season (Q2, Q8)
    agg_words_by_char_episode.csv—words & sentences per character×episode (Q3, Q4, Q7)
    cooccurrence.csv            — character co-occurrence counts per episode (Q6)
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = "."
OUT_DIR = "clean"
TOP_N_CHARACTERS = 30          # keep only the N most-spoken characters
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw files
# ---------------------------------------------------------------------------
print("Loading raw data...")
scripts = pd.read_csv(os.path.join(RAW_DIR, "simpsons_script_lines.csv"), low_memory=False)
episodes = pd.read_csv(os.path.join(RAW_DIR, "simpsons_episodes.csv"))
characters = pd.read_csv(os.path.join(RAW_DIR, "simpsons_characters.csv"))
locations = pd.read_csv(os.path.join(RAW_DIR, "simpsons_locations.csv"))

print(f"  script_lines : {len(scripts):,} rows")
print(f"  episodes     : {len(episodes):,} rows")
print(f"  characters   : {len(characters):,} rows")
print(f"  locations    : {len(locations):,} rows")

# ---------------------------------------------------------------------------
# 2. Clean: episodes
# ---------------------------------------------------------------------------
print("\nCleaning episodes...")
ep = episodes.copy()

# Parse date; drop URL columns (not used in vis)
ep["original_air_date"] = pd.to_datetime(ep["original_air_date"], errors="coerce")
ep = ep.drop(columns=["image_url", "video_url"], errors="ignore")

# Drop the 3 rows that have no imdb_rating (they are unaired/special episodes)
ep = ep.dropna(subset=["imdb_rating"])

# Rename for clarity
ep = ep.rename(columns={"id": "episode_id"})

ep.to_csv(os.path.join(OUT_DIR, "clean_episodes.csv"), index=False)
print(f"  Saved clean_episodes.csv — {len(ep):,} rows")

# ---------------------------------------------------------------------------
# 3. Clean: characters
# ---------------------------------------------------------------------------
print("\nCleaning characters...")
ch = characters.copy()
ch = ch.rename(columns={"id": "character_id"})

# Fill unknown gender
ch["gender"] = ch["gender"].fillna("unknown")

ch.to_csv(os.path.join(OUT_DIR, "clean_characters.csv"), index=False)
print(f"  Saved clean_characters.csv — {len(ch):,} rows")

# ---------------------------------------------------------------------------
# 4. Clean: script_lines
# ---------------------------------------------------------------------------
print("\nCleaning script_lines...")
sl = scripts.copy()

# Keep only actual spoken lines (speaking_line column has stray data in one row)
sl = sl[sl["speaking_line"].astype(str).str.lower() == "true"]

# Drop rows with no character or no spoken words
sl = sl.dropna(subset=["character_id", "spoken_words"])

# Fix dtypes
sl["character_id"] = sl["character_id"].astype(int)
sl["word_count"] = pd.to_numeric(sl["word_count"], errors="coerce").fillna(0).astype(int)

# Drop columns unused in the vis
drop_cols = ["raw_text", "timestamp_in_ms", "normalized_text", "number"]
sl = sl.drop(columns=[c for c in drop_cols if c in sl.columns])

# ---------------------------------------------------------------------------
# 4a. Restrict to main characters (top TOP_N by total word count)
# ---------------------------------------------------------------------------
top_chars = (
    sl.groupby("character_id")["word_count"]
    .sum()
    .nlargest(TOP_N_CHARACTERS)
    .index
)
sl = sl[sl["character_id"].isin(top_chars)]

# ---------------------------------------------------------------------------
# 4b. Enrich: join character name and episode metadata
# ---------------------------------------------------------------------------
sl = sl.merge(
    ch[["character_id", "name", "gender"]],
    on="character_id",
    how="left",
)
sl = sl.merge(
    ep[["episode_id", "season", "number_in_season", "title", "imdb_rating"]],
    on="episode_id",
    how="left",
)

# Drop rows whose episode isn't in the clean episodes table
sl = sl.dropna(subset=["season"])
sl["season"] = sl["season"].astype(int)
sl["number_in_season"] = sl["number_in_season"].astype(int)

# Sentence count: each row = 1 spoken line = 1 sentence
sl["sentence_count"] = 1

# Reorder columns for readability
col_order = [
    "id", "episode_id", "season", "number_in_season", "title",
    "character_id", "name", "gender",
    "location_id", "raw_location_text",
    "spoken_words", "word_count", "sentence_count",
    "imdb_rating",
]
sl = sl[[c for c in col_order if c in sl.columns]]

sl.to_csv(os.path.join(OUT_DIR, "clean_script_lines.csv"), index=False)
print(f"  Saved clean_script_lines.csv — {len(sl):,} rows  ({TOP_N_CHARACTERS} characters)")

# ---------------------------------------------------------------------------
# 5. Aggregations
# ---------------------------------------------------------------------------
print("\nBuilding aggregations...")

# -- Q1 & Q5: total words and sentences per character ----------------------
agg_char = (
    sl.groupby(["character_id", "name", "gender"])
    .agg(
        total_words=("word_count", "sum"),
        total_sentences=("sentence_count", "sum"),
        avg_words_per_sentence=("word_count", "mean"),
    )
    .reset_index()
    .sort_values("total_words", ascending=False)
)
agg_char.to_csv(os.path.join(OUT_DIR, "agg_words_by_char.csv"), index=False)
print(f"  Saved agg_words_by_char.csv — {len(agg_char):,} rows")

# -- Q2 & Q8: words and sentences per character × season -------------------
agg_season = (
    sl.groupby(["character_id", "name", "season"])
    .agg(
        total_words=("word_count", "sum"),
        total_sentences=("sentence_count", "sum"),
        avg_words_per_sentence=("word_count", "mean"),
    )
    .reset_index()
    .sort_values(["name", "season"])
)
agg_season.to_csv(os.path.join(OUT_DIR, "agg_words_by_char_season.csv"), index=False)
print(f"  Saved agg_words_by_char_season.csv — {len(agg_season):,} rows")

# -- Q3, Q4 & Q7: words and sentences per character × episode --------------
agg_episode = (
    sl.groupby(["character_id", "name", "episode_id", "season", "number_in_season", "title", "imdb_rating"])
    .agg(
        total_words=("word_count", "sum"),
        total_sentences=("sentence_count", "sum"),
        avg_words_per_sentence=("word_count", "mean"),
    )
    .reset_index()
    .sort_values(["season", "number_in_season", "name"])
)
agg_episode.to_csv(os.path.join(OUT_DIR, "agg_words_by_char_episode.csv"), index=False)
print(f"  Saved agg_words_by_char_episode.csv — {len(agg_episode):,} rows")

# -- Q6: character co-occurrence per episode --------------------------------
# For each episode, get the set of characters that appear in it,
# then count how many episodes each pair shares.
print("  Building co-occurrence matrix (may take a few seconds)...")

# One row per (episode, character)
ep_char = sl[["episode_id", "season", "character_id", "name"]].drop_duplicates()

# Self-join on episode to get all character pairs per episode
pairs = ep_char.merge(ep_char, on=["episode_id", "season"], suffixes=("_a", "_b"))

# Keep only ordered pairs (a < b) to avoid duplicates and self-pairs
pairs = pairs[pairs["character_id_a"] < pairs["character_id_b"]]

cooccurrence = (
    pairs.groupby(["character_id_a", "name_a", "character_id_b", "name_b"])
    .agg(episode_count=("episode_id", "nunique"))
    .reset_index()
    .sort_values("episode_count", ascending=False)
)
cooccurrence.to_csv(os.path.join(OUT_DIR, "cooccurrence.csv"), index=False)
print(f"  Saved cooccurrence.csv — {len(cooccurrence):,} pairs")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\nPreprocessing complete. Clean files are in:", os.path.abspath(OUT_DIR))

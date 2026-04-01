import re

import pandas as pd
import numpy as np
import re
import os

os.makedirs("data/outputs", exist_ok=True)


df = pd.read_csv("data/simpsons_episodes.csv")


cols_to_drop = ["image_url", "video_url"]
df.drop(columns=cols_to_drop, inplace=True)


# Fix malformed titles (Season 28 rows)

# Three Season-28 titles contain leftover Wikipedia citation
# We strip the trailing quote + bracket reference.


n_fixed = 0
def clean_title(t):
    global n_fixed
    cleaned = re.sub(r'"?\[\d+\]$', '', str(t)).strip()
    if cleaned != t:
        n_fixed += 1
    return cleaned

df["title"] = df["title"].apply(clean_title)
print(f"Titles fixed (citation artifacts removed): {n_fixed}")
assert not df["title"].str.contains(r"\[\d+\]", na=False).any(), \
    "Some titles still contain citation artifacts!"


df["original_air_date"] = pd.to_datetime(df["original_air_date"], errors="coerce")
n_bad_dates = df["original_air_date"].isnull().sum()
print(f"Unparseable dates: {n_bad_dates}")


df["air_year"]         = df["original_air_date"].dt.year
df["air_month"]        = df["original_air_date"].dt.month
df["air_dayofweek_num"] = df["original_air_date"].dt.dayofweek   # 0=Mon … 6=Sun
day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",
           4:"Friday",5:"Saturday",6:"Sunday"}
df["air_dayofweek"] = df["air_dayofweek_num"].map(day_map)


if "original_air_year" in df.columns:
    mismatch = (df["original_air_year"] != df["air_year"]).sum()
    print(f"Year mismatches between raw column and parsed date: {mismatch}")
    df.drop(columns=["original_air_year"], inplace=True)


print("\nMissing data in episodes:")
for col in ["imdb_rating", "imdb_votes", "us_viewers_in_millions", "views"]:
    if col in df.columns:
        print(f"  {col:<25}: {df[col].isnull().sum()}")


df["imdb_votes"] = df["imdb_votes"].astype("Int64")
df["views"]      = df["views"].astype("Int64")


assert df.duplicated(subset=["id"]).sum() == 0, "Duplicate episode IDs found!"
assert df.duplicated(subset=["season","number_in_season"]).sum() == 0, \
    "Duplicate (season, episode) pairs found!"

df.sort_values("original_air_date", inplace=True)
df.reset_index(drop=True, inplace=True)


df["imdb_rating_roll5"] = (
    df["imdb_rating"]
    .rolling(window=5, min_periods=3, center=True)
    .mean()
    .round(3)
)


season_agg = df.groupby("season").agg(
    season_avg_imdb    = ("imdb_rating",            "mean"),
    season_avg_viewers = ("us_viewers_in_millions",  "mean"),
    season_n_episodes  = ("id",                      "count"),
).round(3).reset_index()
df = df.merge(season_agg, on="season", how="left")

#DROP SEASON 28 (only 1 episode, outlier ratings/viewership, and incomplete data)
df = df[df["season"] != 28]

print(f"\nEpisodes dataset final shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes.to_string()}\n")



chars = pd.read_csv("data/simpsons_characters.csv")


print(f"Characters raw shape: {chars.shape}")
print(f"Columns: {chars.columns.tolist()}")


chars.columns = chars.columns.str.strip().str.lower().str.replace(" ", "_")


def fix_char_name(name):
    if isinstance(name, str) and name.isupper():
        return name.title()
    return name

chars["name"] = chars["name"].apply(fix_char_name)


chars["normalized_name"] = chars["normalized_name"].str.strip().str.lower()


gender_map = {"m": "male", "f": "female"}
chars["gender"] = chars["gender"].map(gender_map)   # NaN stays NaN

n_no_gender = chars["gender"].isnull().sum()
print(f"Characters with unknown gender: {n_no_gender} "
      f"({n_no_gender/len(chars)*100:.1f}%)")


assert chars.duplicated(subset=["id"]).sum() == 0, \
    "Duplicate character IDs found!"


print(f"Characters clean shape: {chars.shape}")
print(f"Gender distribution:\n{chars['gender'].value_counts(dropna=False)}\n")



locs = pd.read_csv("data/simpsons_locations.csv")


print(f"Locations raw shape: {locs.shape}")
print(f"Columns: {locs.columns.tolist()}")


locs.columns = locs.columns.str.strip().str.lower().str.replace(" ", "_")


def fix_loc_name(name):
    if isinstance(name, str) and name.isupper():
        return name.title()
    return name

locs["name"] = locs["name"].apply(fix_loc_name)


locs["normalized_name"] = locs["normalized_name"].str.strip().str.lower()


n_dup_names = locs.duplicated(subset=["normalized_name"]).sum()
print(f"Duplicate location names (normalised): {n_dup_names}")
locs.drop_duplicates(subset=["normalized_name"], keep="first", inplace=True)


assert locs.duplicated(subset=["id"]).sum() == 0, \
    "Duplicate location IDs found!"

locs.reset_index(drop=True, inplace=True)
print(f"Locations clean shape: {locs.shape}\n")


episodes_out  = "data/outputs/simpsons_episodes_clean.csv"
chars_out     = "data/outputs/simpsons_characters_clean.csv"
locs_out      = "data/outputs/simpsons_locations_clean.csv"

df.to_csv(episodes_out, index=False)
chars.to_csv(chars_out, index=False)
locs.to_csv(locs_out, index=False)

print("Clean datasets saved:")
print(f"  {episodes_out}  — {df.shape}")
print(f"  {chars_out}     — {chars.shape}")
print(f"  {locs_out}      — {locs.shape}")




INPUT_FILE = "data/simpsons_script_lines.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)




# 2.  DROP FULLY-EMPTY ROWS


n_before = len(df)
df.dropna(how="all", inplace=True)
n_dropped_empty = n_before - len(df)



# 3.  TYPE CASTING


# Integer IDs – use nullable Int64 to survive NaN foreign keys gracefully
for col in ["id", "episode_id", "number", "character_id", "location_id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# Timestamp: keep as nullable float (milliseconds); convert to timedelta helper
df["timestamp_in_ms"] = pd.to_numeric(df["timestamp_in_ms"], errors="coerce")

# word_count: nullable int
df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").astype("Int64")




# 4.  FIX speaking_line FLAG
#     Observed representations: True/False, "true"/"false", 1/0, NaN


def to_bool(val):
    """Coerce various truthy representations to Python bool / pd.NA."""
    if pd.isna(val):
        return pd.NA
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true",  "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return pd.NA

df["speaking_line"] = df["speaking_line"].apply(to_bool).astype("boolean")

n_null_speaking = df["speaking_line"].isna().sum()




# 5.  CLEAN TEXT COLUMNS


CITATION_RE = re.compile(r'\[\d+\]')

def clean_text(val):
    """
    Strip whitespace, collapse runs, remove Wikipedia citation artefacts.
    Returns the cleaned string or pd.NA if the result is empty.
    """
    if pd.isna(val):
        return pd.NA
    s = str(val)
    s = CITATION_RE.sub("", s)        # remove [N] citations
    s = re.sub(r'\s+', ' ', s).strip()  # collapse whitespace
    return s if s else pd.NA

text_cols = ["raw_text", "spoken_words", "normalized_text"]

citation_counts = {}
for col in text_cols:
    if col not in df.columns:
        continue
    # count rows that had a citation before cleaning
    had_citation = df[col].astype(str).str.contains(r'\[\d+\]', na=False).sum()
    citation_counts[col] = had_citation
    df[col] = df[col].apply(clean_text)

print(f"\nCitation artefacts removed per text column:")
for col, n in citation_counts.items():
    print(f"  {col:<20}: {n}")


# 6.  RECOMPUTE & VALIDATE word_count
#     word_count should equal the number of whitespace-delimited tokens
#     in normalized_text.


def count_words(val):
    if pd.isna(val):
        return pd.NA
    tokens = str(val).split()
    return len(tokens) if tokens else pd.NA

df["word_count_recomputed"] = df["normalized_text"].apply(count_words).astype("Int64")


mismatch_mask = (
    df["word_count"].notna()
    & df["word_count_recomputed"].notna()
    & (df["word_count"] != df["word_count_recomputed"])
)
n_mismatch = mismatch_mask.sum()
print(f"\nword_count mismatches (original vs recomputed): {n_mismatch}")


df["word_count"] = df["word_count_recomputed"]
df.drop(columns=["word_count_recomputed"], inplace=True)


# 7.  FIX raw_character_text / raw_location_text
#     ALL-CAPS entries → Title Case  (mirrors characters/locations cleaning)


def fix_name(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    return s.title() if s.isupper() else s

for col in ["raw_character_text", "raw_location_text"]:
    if col in df.columns:
        df[col] = df[col].apply(fix_name)

print("\nraw_character_text / raw_location_text title-cased where ALL-CAPS.")


# 8.  FOREIGN KEY VALIDATION
#     Cross-check against the already-cleaned reference tables.


REF_FILES = {
    "episode_id"   : "data/outputs/simpsons_episodes_clean.csv",
    "character_id" : "data/outputs/simpsons_characters_clean.csv",
    "location_id"  : "data/outputs/simpsons_locations_clean.csv",
}

print("\nForeign key validation:")
for fk_col, ref_path in REF_FILES.items():
    if not os.path.exists(ref_path):
        print(f"  [{fk_col}] SKIP – reference file not found: {ref_path}")
        continue

    ref_ids = pd.read_csv(ref_path, usecols=["id"])["id"].dropna().astype("Int64")
    valid_ids = set(ref_ids)

    fk_vals = df[fk_col].dropna()
    orphans = fk_vals[~fk_vals.isin(valid_ids)]
    pct = len(orphans) / len(fk_vals) * 100 if len(fk_vals) else 0
    print(f"  {fk_col:<15}: {len(orphans):>6} orphan FK values "
          f"({pct:.2f}% of non-null)")


# 9.  DERIVED COLUMNS


# is_speaking: clean boolean alias for the flag
df["is_speaking"] = df["speaking_line"].astype("boolean")

# has_dialogue: True when spoken_words is a non-empty string
df["has_dialogue"] = df["spoken_words"].notna()

# line_length: character count of the spoken words (NaN for non-dialogue lines)
df["line_length"] = df["spoken_words"].apply(
    lambda v: len(str(v)) if pd.notna(v) else pd.NA
).astype("Int64")

# timestamp_in_s: human-readable companion (seconds, 2 d.p.)
df["timestamp_in_s"] = (df["timestamp_in_ms"] / 1000).round(2)

print("\nDerived columns added:")
print("  is_speaking, has_dialogue, line_length, timestamp_in_s")


# 10. SORT & RESET INDEX


df.sort_values(["episode_id", "number"], inplace=True)
df.reset_index(drop=True, inplace=True)


# 11. INTEGRITY ASSERTIONS


assert df.duplicated(subset=["id"]).sum() == 0, \
    "Duplicate script-line IDs found!"

assert df["number"].isnull().mean() < 0.05, \
    "More than 5 % of line numbers are null – investigate!"

assert df["episode_id"].isnull().mean() < 0.01, \
    "More than 1 % of episode_id values are null – investigate!"


# 12. MISSING DATA SUMMARY


print("\nMissing data summary (columns with any nulls):")
null_counts = df.isnull().sum()
for col, n in null_counts[null_counts > 0].items():
    pct = n / len(df) * 100
    print(f"  {col:<25}: {n:>7}  ({pct:.1f}%)")


# 13. FINAL STATS & SAVE


print(f"\nFinal shape : {df.shape}")
print(f"Dtypes:\n{df.dtypes.to_string()}\n")

out_path = "data/outputs/simpsons_script_lines_clean.csv"
df.to_csv(out_path, index=False)
print(f"Clean dataset saved → {out_path}")
print("=" * 60)

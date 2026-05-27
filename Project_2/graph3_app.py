import pandas as pd
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Simpsons Word Distribution", layout="wide")
st.title("Word Distribution Comparison by Character Pair")
st.markdown(
    "Comparison of two characters and how their word count is distributed per line, "
    "across seasons and episodes."
)

df=pd.read_csv("Project_2/clean/graph3_duos.csv")

color_scale = alt.Scale(
    domain=["Homer Simpson", "Marge Simpson", "Bart Simpson", "Lisa Simpson", "Moe Szyslak", "C. Montgomery Burns"],
    range=["#0072B2", "#CC79A7", "#E69F00", "#009E73", "#D55E00", "#000000"],
)

duo_options = ["Homer vs Bart", "Marge vs Bart", "Bart vs Lisa", "Homer vs Moe", "Homer vs Burns"]

with st.sidebar:
    st.header("Filters")
    selected_duo = st.selectbox("Character pair", duo_options)
    df_duo = df[df["duo"] == selected_duo]
    season_options = sorted(df_duo["season"].unique().tolist())
    selected_season = st.selectbox("Season", season_options)
    df_season = df_duo[df_duo["season"] == selected_season]
    episode_options = sorted(df_season["number_in_season"].unique().tolist())
    selected_episode = st.selectbox(
        "Episode (within selected season)", episode_options,
        help="Only episodes that contain both characters are listed.",
    )

df_episode = df_season[df_season["number_in_season"] == selected_episode]

shared_enc = dict(
    x=alt.X("word_count:Q", bin=alt.Bin(step=5), title="Words per line"),
    y=alt.Y("count()", stack=None, title="Number of lines"),
    color=alt.Color("name:N", scale=color_scale, title="Character"),
    tooltip=[
        alt.Tooltip("name:N",       title="Character"),
        alt.Tooltip("word_count:Q", title="Words per line", bin=True),
        alt.Tooltip("count()",      title="Lines"),
    ],
)

combined= (
    alt.Chart(df_season).
    mark_bar(opacity=0.7,stroke="white",strokeWidth=1)
    .encode(**shared_enc)
    .properties(width=400,height=300,title=f"Season {selected_season} full season") 
    | alt.Chart(df_episode).mark_bar(opacity=0.7,stroke="white",strokeWidth=1)
    .encode(**shared_enc)
    .properties(width=400,height=300,title=f"Season {selected_season}, Episode {selected_episode}")).configure_view(stroke=None).configure_axis(grid=False)

st.altair_chart(combined, width='stretch')

col1, col2, col3 = st.columns(3)
col1.metric("Pair",    selected_duo)
col2.metric("Season",  f"Season {selected_season}  ({len(df_season)} lines)")
col3.metric("Episode", f"Ep. {selected_episode}  ({len(df_episode)} lines)")

with st.expander("Design & Accessibility Notes"):
    st.markdown(
        "For `graph3_app.py`, which compares overlapping word-per-line distributions between character pairs across seasons "
        "and episodes, we adopted the Wong colorblind-safe palette throughout, avoiding any red-green combinations. Color "
        "assignments are fixed to named characters regardless of which pair is selected: Homer always maps to #0072B2, Marge "
        "to #CC79A7, Bart to #E69F00, Lisa to #009E73, Moe to #D55E00, and Burns to #000000. This fixed mapping is enforced "
        "through a shared `color_scale` used by both the season-wide and episode-level charts, so the same character always "
        "carries the same color across both views. Overlapping bars use `opacity=0.7` with `stroke='white'` and `strokeWidth=1` "
        "because without the stroke, bars sharing a bin would visually merge into an indistinguishable block; the white border "
        "preserves individual shapes while the 0.7 opacity level ensures neither distribution fully occludes the other. We set "
        "`stack=None` explicitly so bars overlay rather than stack, preserving the actual shape of each character's distribution "
        "rather than collapsing it into a cumulative form. Chart titles are generated dynamically to reflect the current season "
        "and episode context, and axes are labeled 'Words per line' and 'Number of lines' to keep interpretation unambiguous."
    )

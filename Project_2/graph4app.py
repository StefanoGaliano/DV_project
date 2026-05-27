import pandas as pd
import altair as alt
import streamlit as st
 
alt.data_transformers.disable_max_rows()
 
lines = pd.read_csv('data/clean/clean_script_lines.csv')
 
SEASON = 11
EPISODE = 2
EPISODE_TITLE = "Brother's Little Helper"
 
df_ep = lines[(lines['season'] == SEASON) & (lines['number_in_season'] == EPISODE)].copy()
char_options = sorted(df_ep['name'].unique().tolist())
 
st.title('Word Distribution — Episode with Most Characters')
st.markdown(f'**S{SEASON}E{EPISODE} — {EPISODE_TITLE}** · {len(char_options)} unique characters appear in this episode.')
 
col1, col2 = st.columns(2)
char_a = col1.selectbox('Character A', char_options, index=char_options.index('Homer Simpson'))
char_b = col2.selectbox('Character B', char_options, index=char_options.index('Bart Simpson'))
 
if char_a == char_b:
    st.warning('Please select two different characters.')
    st.stop()
 
df_plot = df_ep[df_ep['name'].isin([char_a, char_b])]
 
color_scale = alt.Scale(domain=[char_a, char_b], range=['#0072B2', '#E69F00'])
 
chart = alt.Chart(df_plot).mark_bar(opacity=0.7, stroke='white', strokeWidth=1).encode(
    x=alt.X('word_count:Q', bin=alt.Bin(maxbins=20), title='Words per line'),
    y=alt.Y('count()', stack=None, title='Number of lines'),
    color=alt.Color('name:N', title='Character', scale=color_scale, legend=alt.Legend(orient='left')),
    tooltip=['name:N', alt.Tooltip('word_count:Q', bin=True), 'count()']
).properties(width=650, height=350, title=f'{char_a} vs {char_b}')
 
st.altair_chart(chart, width='stretch')
st.caption(f'Lines shown: {len(df_plot)}')

with st.expander("Design & Accessibility Notes"):
    st.markdown(
        "For `graph4app.py`, which shows overlapping histograms for any two user-selected characters from the single episode "
        "with the highest character count (S11E2, \"Brother's Little Helper\"), we built the color scale dynamically based on "
        "the current selection, always assigning #0072B2 to Character A and #E69F00 to Character B. This is a deliberate "
        "departure from `graph3_app.py`, where colors are fixed to specific named characters: here, because any two characters "
        "from the episode can be chosen, a positional assignment (A vs. B) is more consistent and predictable for the user. "
        "Both colors are from the Wong colorblind-safe palette with no red-green pairing. We applied `opacity=0.7` with "
        "`stroke='white'` and `strokeWidth=1` for the same reason as in graph3, since without the stroke the overlapping "
        "histogram area would appear as a third blended color rather than two distinct distributions. `stack=None` forces "
        "overlay rather than stacking, which matters more here than in the season-wide view because episode-level data is "
        "sparse and stacking would misrepresent the distributions as cumulative. We set `maxbins=20` to match the lower data "
        "volume at episode level. The chart title updates dynamically to name the two selected characters, a caption reports "
        "the total number of lines plotted, and a warning fires if the same character is selected twice."
    )


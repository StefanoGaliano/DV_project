from question_1 import chart_q1_clean as q1_chart
from question_2 import chart_q2 as q2_chart
from question_3 import chart_q3 as q3_chart
from question_4 import chart_q4 as q4_chart
from question_5 import chart_q5 as q5_chart
from question_6 import chart_q6 as q6_chart

import streamlit as st


st.markdown(
    """
    <style>
      .block-container {
        max-width: 1600px;  /* set desired width */
        padding-left: 2rem;
        padding-right: 2rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("The Simpsons: A Data Visualization Journey")

row1 = st.columns(3)
row2 = st.columns(3)
for i, col in enumerate(row1):
    tile = col.container(width=500)
    tile.altair_chart([q1_chart, q2_chart, q3_chart][i], use_container_width=True)
st.space("xxlarge")
for i, col in enumerate(row2):
    tile = col.container(width=500)
    tile.altair_chart([q4_chart, q5_chart, q6_chart][i], use_container_width=True)

st.write("What")
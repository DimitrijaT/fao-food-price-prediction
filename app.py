from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
from prediction import predict

"""
# Welcome to Streamlit!

Edit `/app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)
#
#     Point = namedtuple('Point', 'x y')
#     data = []
#
#     points_per_turn = total_points / num_turns
#
#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))
#
#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')

st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

if st.button("Predict type of Iris"):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])



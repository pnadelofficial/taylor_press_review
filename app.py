import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from utils import TFIDF
import nltk

nltk.download('punkt_tab')

st.title("Press Review viewer")

@st.cache_resource
def read_data():
    articles = pd.read_csv("./final_press_review.csv")
    articles['time_frame'] = pd.to_datetime(articles['Date'], format='mixed')
    return articles# .drop_duplicates(subset="Title")
articles = read_data()

if "selected_newspapers" not in st.session_state:
    st.session_state.selected_newspapers = []

newspapers = st.multiselect(
    "Pick by newspaper",
    articles.Newspaper.unique(),
    default=st.session_state.selected_newspapers,
    key="newspaper_widget" 
)

if len(newspapers) > 0:
    data = []
    for newspaper in newspapers:
        subset = articles[articles.Newspaper == newspaper]
        subset_yearly = subset.groupby(subset['time_frame'].dt.to_period('Y')).size()
        data.append(go.Bar(name=newspaper, x=subset_yearly.index.year, y=subset_yearly.values))
    fig = go.Figure(data=data)
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points")

    if len(event['selection']['points']) > 0:
        selected_year = math.ceil(event['selection']['points'][0]["x"])
        amount = event['selection']['points'][0]["y"]
        selected_year = pd.to_datetime(f"1/1/{selected_year}")
        st.write(f"**Year selected**: {selected_year.strftime('%Y')}")
        tabs = st.tabs(newspapers)
        for i, tab in enumerate(tabs):
            with tab:
                newspaper = newspapers[i]
                subset = articles[(articles.Newspaper == newspaper) & (articles.time_frame.dt.year == selected_year.year)].reset_index(drop=True)
                search_term = st.text_input("Search for a term", key=f"search_{newspaper}")
                if search_term.strip():
                    tfidf = TFIDF(subset['chunks'].tolist())
                    results = tfidf.query(search_term)
                    subset['similarity'] = 0
                    for idx, score in results:
                        subset.at[idx, 'similarity'] = score
                    subset = subset.sort_values('similarity', ascending=False)
                    subset = subset[subset['similarity'] > 0]

                st.write(f"**Articles for {newspaper}**: {len(subset)}")
                for i, row in subset.iterrows():
                    with st.expander(f"*{row['Title']}* by {row['Author']} - {row['time_frame'].strftime('%B %d, %Y')}"):
                        text = row['chunks'].replace("`", r"\`").replace("$", r"\$")
                        st.write(f"{text}")

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

st.title("Press Review viewer")

@st.cache_resource
def read_data():
    df = pd.read_csv("./chunked_press_review.csv")
    df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.replace(" 00:00:00", "")))
    articles = df.groupby(["Title", "Date", "Author", "Newspaper", "Source", "newspaper_type"])['chunks'].apply(lambda x: " ".join(x).replace("  ", " ")).reset_index()
    articles = articles.sort_values(["Date"])
    articles['time_frame'] = pd.to_datetime(articles['Date'])
    articles['Title'] = articles["Title"].apply(lambda x: x.strip())
    return articles.drop_duplicates(subset="Title")
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
        selected_year = math.floor(event['selection']['points'][0]["x"])
        amount = event['selection']['points'][0]["y"]
        selected_year = pd.to_datetime(f"1/1/{selected_year}")
        st.write(f"**Year selected**: {selected_year.strftime('%Y')}")
        tabs = st.tabs(newspapers)
        for i, tab in enumerate(tabs):
            with tab:
                newspaper = newspapers[i]
                search_term = st.text_input("Search for a term")
                if search_term.strip():
                    print(search_term)
                    print(articles.chunks)
                    subset = articles[(articles.Newspaper == newspaper) & (articles.time_frame.dt.year == selected_year.year) & (articles.chunks.str.match(search_term))]
                    print(subset)
                else:                
                    subset = articles[(articles.Newspaper == newspaper) & (articles.time_frame.dt.year == selected_year.year)]
                st.write(f"**Articles for {newspaper}**: {len(subset)}")
                for i, row in subset.iterrows():
                    with st.expander(f"*{row['Title']}* by {row['Author']} - {row['time_frame'].strftime('%B %d, %Y')}"):
                        text = row['chunks'].replace("`", r"\`").replace("$", r"\$")
                        st.write(f"{text}")

# DO THESE THINGS FOR NEXT TIME
# add search bar in the tab
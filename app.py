import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from utils import TFIDF, search_snippet
import nltk

nltk.download('punkt_tab')

st.title("Press Review viewer")

@st.cache_resource
def read_data():
    articles = pd.read_csv("./all_ratings731.csv")
    def parse_date(val):
        for fmt in ('%Y-%m-%d', '%B %d, %Y'):
            try:
                return pd.to_datetime(val, format=fmt)
            except (ValueError, TypeError) as e:
                continue
        return pd.NaT  # fallback if nothing matches

    articles['time_frame'] = articles['Date'].apply(lambda x: x.replace(" 00:00:00", "")).apply(parse_date)
    whitelist_titles = {'No Title found', "No title found"}
    whitelisted_df = articles[articles['Title'].isin(whitelist_titles)]
    normal_df = articles[~articles['Title'].isin(whitelist_titles)]
    result = pd.concat([normal_df.drop_duplicates(subset='Title'), whitelisted_df]).sort_index()
    return result
articles = read_data()

if "selected_newspapers" not in st.session_state:
    st.session_state.selected_newspapers = []

if "expand_all" not in st.session_state:
    st.session_state.expand_all = False

def toggle_expand():
    st.session_state.expand_all = not st.session_state.expand_all

time_frame = st.selectbox(
    "Pick a time frame",
    ["By year", "By month"]
)

if time_frame == "By year":
    newspapers = st.multiselect(
        "Pick by newspaper",
        articles.Newspaper.unique(),
        default=st.session_state.selected_newspapers,
        key="newspaper_widget1" 
    )

if time_frame == "By month":
    year = st.selectbox(
        "Pick a year",
        articles['time_frame'].dt.year.dropna().sort_values().unique()
    )
    newspapers = st.multiselect(
        "Pick by newspaper",
        articles.Newspaper.unique(),
        default=st.session_state.selected_newspapers,
        key="newspaper_widget2" 
    )

if len(newspapers) > 0:
    data = []
    for newspaper in newspapers:
        subset = articles[articles.Newspaper == newspaper]
        if time_frame == "By year":
            subset['Year'] = subset['time_frame'].dt.to_period('Y')
            subset_yearly = subset.groupby("Year").agg(
                weighted_count=('rating', 'sum'),
                article_count=('rating', 'size')
            )
            subset_yearly['normalized'] = (subset_yearly['weighted_count'] / subset_yearly['article_count'])*5 # not sure if I want this yet
            data.append(go.Bar(name=newspaper, x=subset_yearly.index.year, y=subset_yearly['normalized'], customdata=subset_yearly['article_count'], hovertemplate='Year: %{x}<br>Normalized relevance: %{y:.3f}<br>Article count: %{customdata}<extra></extra>'))
        else:
            subset = subset[(subset.time_frame.dt.year == year)]
            subset['Month'] = subset['time_frame'].dt.to_period('M')
            subset_monthly = subset.groupby("Month").agg(
                weighted_count=('rating', 'sum'),
                article_count=('rating', 'size')
            )
            subset_monthly['normalized'] = (subset_monthly['weighted_count'] / subset_monthly['article_count'])*5
            data.append(go.Bar(name=newspaper, x=subset_monthly.index.to_timestamp(), y=subset_monthly['normalized'], customdata=subset_monthly['article_count'], hovertemplate='Month: %{x|%B %Y}<br>Normalized relevance: %{y:.3f}<br>Article count: %{customdata}<extra></extra>'))
    if data[0]['x'].size == 0:
        st.warning("No data available for the selected newspapers and time frame. Please choose another configuration.")
        st.stop()
    fig = go.Figure(data=data)
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points")

    if len(event['selection']['points']) > 0:
        if isinstance(event['selection']['points'][0]["x"], str):
            selected_month = pd.to_datetime(event['selection']['points'][0]["x"])
            st.write(f"**Month selected**: {selected_month.strftime('%B %Y')}, {year}")
            amount = event['selection']['points'][0]["y"]
            tabs = st.tabs(newspapers)
            for i, tab in enumerate(tabs):
                with tab:
                    newspaper = newspapers[i]
                    subset = articles[(articles.Newspaper == newspaper) & (articles.time_frame.dt.to_period('M') == selected_month.to_period('M'))].reset_index(drop=True)
                    search_term = st.text_input("Search for a term", key=f"search_{newspaper}")
                    context_size = st.number_input(
                        "Context size (sentences before and after the match)",
                        min_value=0,
                        value=1,
                        step=1,
                        key=f"context_{newspaper}",
                    )
                    if search_term.strip():
                        tfidf = TFIDF(subset['chunks'].tolist())
                        results = tfidf.query(search_term)
                        subset['similarity'] = 0.0
                        for idx, score in results:
                            subset.at[idx, 'similarity'] = score
                        subset = subset.sort_values('similarity', ascending=False)
                        subset = subset[subset['similarity'] > 0.0]

                    st.write(f"**Articles for {newspaper}**: {len(subset)}")
                    st.button(
                        "Collapse All" if st.session_state.expand_all else "Expand All",
                        on_click=toggle_expand,
                        key=f"toggle_{newspaper}"
                    )
                    for i, row in subset.iterrows():
                        with st.expander(f"*{row['Title']}* by {row['Author']} - {row['time_frame'].strftime('%B %d, %Y')}", expanded=st.session_state.expand_all):
                            if search_term.strip():
                                snippet = search_snippet(row['chunks'], search_term, context_size)
                                st.markdown(snippet, unsafe_allow_html=True)
                            else:
                                text = row['chunks'].replace("`", r"\`").replace("$", r"\$")
                                st.write(f"{text}")
        else:
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
                    context_size = st.number_input(
                        "Context size (sentences before and after the match)",
                        min_value=0,
                        value=1,
                        step=1,
                        key=f"context_{newspaper}",
                    )
                    if search_term.strip():
                        tfidf = TFIDF(subset['chunks'].tolist())
                        results = tfidf.query(search_term)
                        subset['similarity'] = 0.0
                        for idx, score in results:
                            subset.at[idx, 'similarity'] = score
                        subset = subset.sort_values('similarity', ascending=False)
                        subset = subset[subset['similarity'] > 0.0]

                    st.write(f"**Articles for {newspaper}**: {len(subset)}")
                    st.button(
                        "Collapse All" if st.session_state.expand_all else "Expand All",
                        on_click=toggle_expand,
                        key=f"toggle_{newspaper}"
                    )
                    for i, row in subset.iterrows():
                        with st.expander(f"*{row['Title']}* by {row['Author']} - {row['time_frame'].strftime('%B %d, %Y')}", expanded=st.session_state.expand_all):
                            if search_term.strip():
                                snippet = search_snippet(row['chunks'], search_term, context_size)
                                st.markdown(snippet, unsafe_allow_html=True)
                            else:
                                text = row['chunks'].replace("`", r"\`").replace("$", r"\$")
                                st.write(f"{text}")

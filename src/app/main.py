from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
import os
from dataclasses import dataclass
import sys
src_in_path = False
for path in sys.path:
    if path.endswith('src'):
        src_in_path = True
if not src_in_path:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
from google_trends.TrendFetcher import TrendFetcher



@dataclass
class UserInput:
    keywords: List[str]
    timeframe: str
    geo: str
    gprop: str


# List of Google properties
GPROP_OPTIONS: Dict[str, str] = dict(zip(['search', 'news', 'images', 'youtube', 'shopping'], [
                                     '', 'news', 'images', 'youtube', 'froogle']))


COUNTRIES: Dict[str, str] = {"Worldwide": "",
                             "Israel": "IL", "United States": "US"}


def get_keywords_from_input() -> List[str]:
    keywords_input = st.text_input("Enter Keywords (comma-separated):")
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    return keywords


def get_timeframe_from_input() -> str:
    timeframe = st.date_input("Select Time Frame", [])
    timeframe = " ".join(map(str, timeframe))
    return timeframe


def get_geo_from_input() -> str:
    geo = st.selectbox("Select Location", list(COUNTRIES.keys()))
    return COUNTRIES[geo]


def get_gprop_from_input() -> str:
    gprop = st.selectbox("Select Google Property", list(GPROP_OPTIONS.keys()))
    return GPROP_OPTIONS[gprop]


def init_screen() -> UserInput:
    st.title("Google Trends Explorer")
    keywords = get_keywords_from_input()
    timeframe = get_timeframe_from_input()
    geo = get_geo_from_input()
    gprop = get_gprop_from_input()
    return UserInput(
        keywords=keywords,
        timeframe=timeframe,
        geo=geo,
        gprop=gprop
    )


def fetch_trends_data(user_input: UserInput) -> Optional[pd.DataFrame]:
    fetcher = TrendFetcher(
        keywords=user_input.keywords,
        timeframe=user_input.timeframe,
        geo=user_input.geo,
        gprop=user_input.gprop
    )
    return fetcher.fetch()


def handle_user_input(user_input: UserInput):
    if user_input.keywords and user_input.timeframe:
        trends_data = fetch_trends_data(user_input)
        if trends_data is not None:
            st.session_state['trends_data'] = trends_data
        else:
            st.warning("No data returned for the selected parameters.")
    else:
        st.warning("Please enter valid keywords and select a timeframe.")

def display_trends_csv():
    st.write("Trends Data:")
    st.dataframe(st.session_state['trends_data'])

def download_trends_csv():
    csv = st.session_state['trends_data'].to_csv().encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="trends_data.csv",
        mime="text/csv"
    )

def handle_trends_csv():
    display_trends_csv()
    download_trends_csv()


def display_trends(user_input: UserInput):
    if 'trends_data' in st.session_state:
        handle_trends_csv()
        # Plotting trend graphs
        for keyword in user_input.keywords:
            if keyword in st.session_state['trends_data'].columns:
                st.subheader(keyword)
                st.line_chart(
                    st.session_state['trends_data'][keyword], use_container_width=True)


def main():

    user_input = init_screen()

    # Fetch trends button
    if st.button("Fetch Trends"):
        trends_data = handle_user_input(user_input)

    # Save to CSV option (always visible if data exists)
    display_trends(user_input)


if __name__ == "__main__":
    main()

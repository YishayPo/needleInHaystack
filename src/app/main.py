from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
import os
from dataclasses import dataclass
import sys

src_in_path = False
for path in sys.path:
    if path.endswith("src"):
        src_in_path = True
if not src_in_path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google_trends.TrendFetcher import TrendFetcher
import plotly.graph_objs as go


@dataclass
class UserInput:
    keywords: List[str]
    timeframe: str
    geo: str
    gprop: str
    is_relative_intrest: bool


# List of Google properties
GPROP_OPTIONS: Dict[str, str] = dict(
    zip(
        ["search", "news", "images", "youtube", "shopping"],
        ["", "news", "images", "youtube", "froogle"],
    )
)


COUNTRIES: Dict[str, str] = {"Worldwide": "", "Israel": "IL", "United States": "US"}


def get_keywords_from_input() -> List[str]:
    uploaded_file = st.file_uploader(
        "Upload Keywords File (.txt or .csv)", type=["txt", "csv"]
    )
    keywords = []
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            keywords = [line.strip() for line in content.splitlines() if line.strip()]
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
            # Flatten all values into a single list
            keywords = [str(item).strip() for sublist in df.values.tolist() for item in sublist if str(item).strip()]
        st.success(f"Loaded {len(keywords)} keywords from file.")
    else:
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
    is_relative_intrest = st.toggle("Use relative interest?", value=False)

    return UserInput(
        keywords=keywords,
        timeframe=timeframe,
        geo=geo,
        gprop=gprop,
        is_relative_intrest=is_relative_intrest,
    )


def fetch_trends_data(user_input: UserInput) -> Optional[pd.DataFrame]:
    fetcher = TrendFetcher(
        keywords=user_input.keywords,
        timeframe=user_input.timeframe,
        geo=user_input.geo,
        gprop=user_input.gprop,
    )
    return fetcher.fetch(is_relative_intrest=user_input.is_relative_intrest)


def handle_user_input(user_input: UserInput):
    if user_input.keywords and user_input.timeframe:
        trends_data = fetch_trends_data(user_input)
        if trends_data is not None:
            st.session_state["trends_data"] = trends_data
        else:
            st.warning("No data returned for the selected parameters.")
    else:
        st.warning("Please enter valid keywords and select a timeframe.")


def display_trends_csv():
    st.write("Trends Data:")
    st.dataframe(st.session_state["trends_data"])


def create_weekly_trends_data(trends_data: pd.DataFrame) -> bytes:
    df_weekly = trends_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_weekly.index):
        df_weekly.index = pd.to_datetime(df_weekly.index)
    df_weekly = df_weekly.resample("W").mean()
    csv = df_weekly.to_csv().encode("utf-8")
    return csv


def download_trends_csv():
    weekly = st.checkbox("Weekly Data", value=False)
    trends_data = st.session_state["trends_data"]
    if weekly:
        csv = create_weekly_trends_data(trends_data)
    else:
        csv = trends_data.to_csv().encode("utf-8")
    st.download_button(
        label="Download CSV", data=csv, file_name="trends_data.csv", mime="text/csv"
    )


def handle_trends_csv():
    display_trends_csv()
    download_trends_csv()


def display_joined_trends(trends_data: pd.DataFrame, user_input: UserInput):
    if not trends_data.empty:
        fig = go.Figure()
        for keyword in user_input.keywords:
            if keyword in trends_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trends_data.index,
                        y=trends_data[keyword],
                        mode="lines",
                        name=keyword,
                    )
                )
        fig.update_layout(
            title="All Trends",
            xaxis_title="Date",
            yaxis_title="Interest",
            legend_title="Keywords",
        )
        st.plotly_chart(fig, use_container_width=True)


def display_trends(user_input: UserInput):
    if "trends_data" in st.session_state:
        handle_trends_csv()
        trends_data = st.session_state["trends_data"]
        # Plot all trends together with interactive legend
        display_joined_trends(trends_data, user_input)
        # Plot each trend separately
        for keyword in user_input.keywords:
            if keyword in trends_data.columns:
                st.subheader(keyword)
                st.line_chart(trends_data[keyword], use_container_width=True)


def main():

    user_input = init_screen()

    # Fetch trends button
    if st.button("Fetch Trends"):
        handle_user_input(user_input)

    # Save to CSV option (always visible if data exists)
    display_trends(user_input)


if __name__ == "__main__":
    main()

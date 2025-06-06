from typing import Dict
import streamlit as st
import pandas as pd
import os
import sys
src_in_path = False
for path in sys.path:
    if path.endswith('src'):
       src_in_path = True
if not src_in_path: 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google_trends.TrendFetcher import TrendFetcher



# List of Google properties
GPROP_OPTIONS: Dict[str, str] = dict(zip(['search', 'news', 'images', 'youtube', 'shopping'], [
                                     '', 'news', 'images', 'youtube', 'froogle']))

# List of countries for geo selection
# COUNTRIES = [
#     "Worldwide", "IL", "US", "CA", "GB", "AU", "DE", "FR", "IN", "JP", "BR", "CN", "RU", "IT", "ES", "MX", "NL", "SE", "NO", "DK", "FI"
# ]

COUNTRIES = ["Worldwide", "US", "IL"]


def main():
    st.title("Google Trends Explorer")

    # Keyword input
    keywords_input = st.text_input("Enter Keywords (comma-separated):")
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    # Timeframe selection
    timeframe = st.date_input("Select Time Frame", [])

    # Geo selection
    geo = st.selectbox("Select Location", COUNTRIES)

    # Gprop selection
    gprop = st.selectbox("Select Google Property", GPROP_OPTIONS.keys())

    # Fetch trends button
    if st.button("Fetch Trends"):
        if keywords and timeframe:
            fetcher = TrendFetcher(
                keywords=keywords,
                timeframe=" ".join(map(str, timeframe)),
                geo=geo if geo != "Worldwide" else "",
                gprop=GPROP_OPTIONS[gprop]
            )
            trends_data = fetcher.fetch()
            if trends_data is not None:
                # Save to session state
                st.session_state['trends_data'] = trends_data
            else:
                st.warning("No data returned for the selected parameters.")
        else:
            st.warning("Please enter valid keywords and select a timeframe.")

    # Save to CSV option (always visible if data exists)
    if 'trends_data' in st.session_state:
        st.write("Trends Data:")
        st.dataframe(st.session_state['trends_data'])
        csv = st.session_state['trends_data'].to_csv(
            index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="trends_data.csv",
            mime="text/csv"
        )
        # if st.button("Save to CSV"):
        #     st.session_state['trends_data'].to_csv("trends_data.csv")
        #     st.success("Data saved to trends_data.csv")
        # Plotting trend graphs
        for keyword in keywords:
            if keyword in st.session_state['trends_data'].columns:
                st.subheader(keyword)
                st.line_chart(
                    st.session_state['trends_data'][keyword], use_container_width=True)


if __name__ == "__main__":
    main()

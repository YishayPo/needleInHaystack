from dataclasses import dataclass
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import time
from typing import Callable, List, Optional, Union
from logging import log, warning, error, info
import pandas as pd
from enum import Enum
import datetime
import re

KeyWord = str
KeyWords = List[str]
RATE_LIMIT_EXCEED = "429"


@dataclass
class LogLevel:
    DEBUG: int = 10
    INFO: int = 20
    WARNING: int = 30
    ERROR: int = 40
    CRITICAL: int = 50


def no_data_returned_msg(keywords: KeyWords, timeframe: str) -> str:
    return f"No data returned for {keywords} over '{timeframe}'."


def rate_limit_exceeded_msg(
    attempt: int, max_retries: int, backoff: int, error: str
) -> str:
    return f"Rate limited ({RATE_LIMIT_EXCEED}). Retry {attempt}/{max_retries} after {backoff}s. Error: {error}"


def max_retries_reached_msg(max_retries: int) -> str:
    return f"Failed to fetch after {max_retries} retries due to repeated {RATE_LIMIT_EXCEED} errors."


def is_partial_data_msg(keywords: KeyWords, num_partial: int) -> str:
    return f"Data for {keywords} contains {num_partial} partial entries. This may affect analysis."


def error_fetching_data_msg(keywords: KeyWords, msg: str) -> str:
    return f"Error fetching data for {keywords}: {msg}"


@dataclass
class Messages:
    RATE_LIMIT_EXCEED: Callable = rate_limit_exceeded_msg
    NO_DATA_RETURNED: Callable = no_data_returned_msg
    MAX_RETRIES: Callable = max_retries_reached_msg
    PARTIAL_DATA: Callable = is_partial_data_msg
    ERROR_FETCHING_DATA: Callable = error_fetching_data_msg


class TrendFetcher:
    def __init__(
        self,
        keywords: Union[KeyWords, KeyWord],
        timeframe: str = "today 12-m",
        geo: str = "",
        gprop: str = "",
        max_retries: int = 3,
    ):
        """
        Args:
            keywords (str or list of str): Single term or list of terms to query.
            timeframe (str): Time period (e.g. "2023-01-01 2023-06-01" or "today 12-m").
            geo (str): Country code (e.g. "US", "IL"); empty string for worldwide.
            gprop (str): Category (e.g. "news", "images", "youtube"); empty for web search.
            max_retries (int): Number of attempts if a 429 (rate-limit) is encountered.
        """
        self._keywords = [""]
        self._timeframe = ""
        self._geo = ""
        self._gprop = ""
        self._max_retries = 1
        self.keywords = keywords
        self.timeframe = timeframe
        self.geo = geo
        self.gprop = gprop
        self.max_retries = max_retries

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @staticmethod
    def _is_valid_datetime(date_str: str) -> bool:
        """
        Valid date formats (white space is ignored):
        - any valid datetime string recognized by pandas
        - "1-d", "1-m", "1-y" (e.g., "1-d" for 1 day, "1-m" for 1 month, "1-y" for 1 year)
        """
        if re.fullmatch(r"\d+\s*-\s*[dmy]", date_str.strip()):
            return True
        try:
            pd.to_datetime(date_str, errors="raise")
            return True
        except ValueError:
            return False

    @timeframe.setter
    def timeframe(self, value: str):
        """
        Set the timeframe for the trend data.
        """
        assert isinstance(value, str), "Timeframe must be a string."
        assert value.strip(), "Timeframe cannot be empty."
        assert (
            " " in value
        ), "Timeframe must contain a space to separate start and end dates."
        assert (
            len(value.split(" ")) == 2
        ), "Timeframe must contain exactly two parts: start and end dates."
        first_part, second_part = value.split(" ")
        assert self._is_valid_datetime(first_part), f"Invalid start date: {first_part}"
        assert self._is_valid_datetime(second_part), f"Invalid end date: {second_part}"
        self._timeframe = value

    @property
    def geo(self) -> str:
        return self._geo

    @staticmethod
    def _is_valid_geo(geo: str) -> bool:
        """
        Verifies that the geo string is valid.
        Accepts:
        - '' (empty string for worldwide)
        - 2-letter country codes (e.g., 'US', 'IL')
        - 3-letter region codes (e.g., 'US-CA')
        - 3-letter city codes (e.g., 'US-CA-1234')
        """
        if geo == "":
            return True
        # Country code: 2 uppercase letters
        if re.fullmatch(r"[A-Z]{2}", geo):
            return True
        # Region code: 2 uppercase letters, hyphen, 1+ uppercase letters
        if re.fullmatch(r"[A-Z]{2}-[A-Z]+", geo):
            return True
        # City code: 2 uppercase letters, hyphen, 1+ uppercase letters, hyphen, 1+ digits
        if re.fullmatch(r"[A-Z]{2}-[A-Z]+-\d+", geo):
            return True
        return False

    @geo.setter
    def geo(self, value: str):
        assert isinstance(value, str), "Geo must be a string."
        assert self._is_valid_geo(value), f"Invalid geo code: {value}"
        self._geo = value.strip()

    @property
    def keywords(self) -> KeyWords:
        return self._keywords

    @staticmethod
    def _is_list_of_non_empty_strings(lst: List[str]) -> bool:
        if not isinstance(lst, list) or len(lst) == 0:
            return False
        return all(isinstance(item, str) and item.strip() for item in lst)

    @keywords.setter
    def keywords(self, value: Union[KeyWords, KeyWord]):
        """
        Verify that keywords is a non-empty string or a list of non-empty strings.
        """
        msg = f"Invalid keywords: {value}. Must be a non-empty string or a list of non-empty strings."
        assert isinstance(value, (str, list)), msg
        assert len(value) > 0, msg
        if isinstance(value, str):
            value = [value.strip()]
        else:
            assert self._is_list_of_non_empty_strings(value), msg
            value = [k.strip() for k in value]
        self._keywords = value

    @property
    def gprop(self) -> str:
        return self._gprop

    @gprop.setter
    def gprop(self, value: str):
        """
        Set the Google property for the trend data.
        Valid values are:
        - '' (empty string for web search)
        - 'news' for news search
        - 'images' for image search
        - 'youtube' for YouTube search
        - 'froogle' for product search
        """
        valid_gprops = ["", "news", "images", "youtube", "froogle"]
        assert isinstance(value, str), "Gprop must be a string."
        assert (
            value in valid_gprops
        ), f"Invalid gprop: {value}. Must be one of {valid_gprops}."
        self._gprop = value.strip()

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int):
        assert isinstance(value, int), "Max retries must be an integer."
        assert value >= 1, "Max retries must be non-negative."
        self._max_retries = value

    @staticmethod
    def _get_local_offset() -> int:
        """
        Get the local timezone offset in minutes.
        Returns 0 if the timezone is UTC.
        """
        now = datetime.datetime.now()
        local_offset = now.utcoffset()
        if local_offset is None:
            return 0
        return int(local_offset.total_seconds() / 60)

    def __handle_isPartial_column(self, df: pd.DataFrame):
        if "isPartial" in df.columns:
            num_partial = df["isPartial"].sum()
            if num_partial > 0:
                warning(Messages.PARTIAL_DATA(self.keywords, num_partial))

    def __handle_no_data(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            warning(Messages.NO_DATA_RETURNED(self.keywords, self.timeframe))
            return pd.DataFrame()
        else:
            return df

    @staticmethod
    def _add_keyword_df_to_merge_df(
        df: Optional[pd.DataFrame], keyword_df: pd.DataFrame
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return keyword_df
        if keyword_df is None or keyword_df.empty:
            return df
        if "isPartial" in keyword_df.columns:
            if "isPartial" in df.columns:
                df["isPartial"] = df["isPartial"] | keyword_df["isPartial"]
            else:
                df["isPartial"] = keyword_df["isPartial"]
            keyword_df = keyword_df.drop(columns=["isPartial"], errors="ignore")
        return pd.concat([df, keyword_df], axis=1)

    def __fetch_keywords(
        self, pytrend: TrendReq, keywords: Optional[KeyWords] = None
    ) -> pd.DataFrame:
        if keywords is None:
            keywords = self.keywords
        pytrend.build_payload(
            kw_list=keywords, timeframe=self.timeframe, geo=self.geo, gprop=self.gprop
        )
        df = pytrend.interest_over_time()
        return df

    def _fetch_without_retry(
        self, pytrend: TrendReq, is_relative_intrest: bool = False
    ) -> pd.DataFrame:
        if is_relative_intrest:
            df = self.__fetch_keywords(pytrend)
        else:
            df = None
            for keyword in self.keywords:
                keyword_df = self.__fetch_keywords(pytrend, [keyword])
                df = self._add_keyword_df_to_merge_df(df, keyword_df)

        df = self.__handle_no_data(df)
        self.__handle_isPartial_column(df)
        return df.drop(columns=["isPartial"], errors="ignore")

    def fetch(self, is_relative_intrest: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch Google Trends data for the specified keywords.
        Returns a DataFrame with interest over time, or None if no data is returned.
        Retries on HTTP 429 (rate limit exceeded).
        """
        pytrend = TrendReq(hl="en-US", tz=self._get_local_offset())
        # setup for retry logic
        attempt = 0
        backoff = 1

        while attempt < self.max_retries:
            try:
                return self._fetch_without_retry(pytrend, is_relative_intrest)
            except ResponseError as e:
                msg = str(e).lower()
                if RATE_LIMIT_EXCEED in msg:
                    attempt += 1
                    info(
                        Messages.RATE_LIMIT_EXCEED(
                            attempt, self.max_retries, backoff, e
                        )
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    error(Messages.ERROR_FETCHING_DATA(self.keywords, msg))
                    return None
        error(Messages.MAX_RETRIES(self.max_retries))
        return None

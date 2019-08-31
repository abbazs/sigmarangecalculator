#%% Import dutil
import pandas as pd
from datetime import datetime, date, timedelta
from dateutil import parser
from dateutil.relativedelta import TH, relativedelta
from pandas.core.series import Series
import numpy as np

def process_date(input_date):
    if isinstance(input_date, datetime):
        return datetime.combine(input_date, datetime.min.time())
    elif isinstance(input_date, date):
        return datetime.combine(input_date, datetime.min.time())
    elif isinstance(input_date, str):
        return parser.parse(input_date)
    elif isinstance(input_date, Series):
        return input_date.iloc[0]
    elif isinstance(input_date, np.datetime64):
        return input_date
    else:
        print(
            f"input_date data type {type(input_date)} is not yet handled by this function"
        )
        return None


def get_current_date():
    if datetime.today().hour < 17:
        dt = date.today() - timedelta(days=1)
        return datetime.combine(dt, datetime.min.time())
    else:
        return datetime.combine(date.today(), datetime.min.time())


def fix_start_and_end_date(start_date, end_date):
    if end_date is None:
        end_date = start_date
    else:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
    return start_date, end_date


def get_last_month_last_TH(input_date=None):
    if input_date == None:
        lm = datetime.combine(
            date.today().replace(day=1), datetime.min.time()
        ) - timedelta(days=1)
    elif process_date(input_date) == None:
        lm = datetime.combine(
            date.today().replace(day=1), datetime.min.time()
        ) - timedelta(days=1)
    else:
        lm = process_date(input_date).replace(day=1) - timedelta(days=1)
    lt = lm + relativedelta(weekday=TH(-1))
    return lt


def get_dates_360d_split_from_start(start):
    end = date.today()
    df = pd.DataFrame(
        pd.date_range(start=start, end=end, freq="360D"), columns=["START"]
    )
    df = df.assign(END=df["START"].shift(-1) - pd.DateOffset(days=1))
    df["END"].iloc[-1] = datetime.fromordinal(end.toordinal())
    return df


def last_month_last_TH():
    lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(
        days=1
    )
    lt = lm + relativedelta(weekday=TH(-1))
    return lt


def next_month_last_TH():
    lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(
        days=1
    )
    lt = lm + relativedelta(weekday=TH(4))
    return lt


def next_n_thursdays(nths=10):
    return pd.date_range(start=date.today(), periods=nths, freq="W-THU").tolist()


def all_thursdays_between_dates(start, end):
    st = process_date(start)
    nd = process_date(end)
    st, nd = fix_start_and_end_date(st, nd)
    return pd.Series(pd.date_range(start=st, end=nd, freq="W-THU"))


def last_TH_of_months_between_dates(start, end):
    st = process_date(start)
    nd = process_date(end)
    st, nd = fix_start_and_end_date(st, nd)
    df = pd.DataFrame(
        pd.date_range(
            start=st, end=nd, freq="M", normalize=True, closed="right"
        ).append(pd.date_range(start=nd, freq="M", periods=1)),
        columns=["MEXP"],
    )
    return df["MEXP"].apply(lambda x: x + relativedelta(weekday=TH(-1)))

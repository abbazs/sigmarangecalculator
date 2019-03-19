#%% Import dutil
from datetime import datetime, date, timedelta
from dateutil import parser
from dateutil.relativedelta import TH, relativedelta
from pandas.core.series import Series
import pandas as pd

def process_date(input_date):
    if isinstance(input_date, datetime):
        return datetime.combine(input_date, datetime.min.time())
    elif isinstance(input_date, date):
        return datetime.combine(input_date, datetime.min.time())
    elif isinstance(input_date, str):
        return parser.parse(input_date)
    elif isinstance(input_date, Series):
        return input_date.iloc[0]
    else:
        print(f'input_date data type {type(input_date)} is not yet handled by this function')
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
        lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    elif process_date(input_date) == None:
        lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    else:
        lm = process_date(input_date).replace(day=1) - timedelta(days=1)
    lt = lm + relativedelta(weekday=TH(-1))
    return lt

def get_dates_360d_split_from_start(start):
    end = date.today()
    df = pd.DataFrame(pd.date_range(start=start, end=end, freq='360D'), columns=['START'])
    df = df.assign(END=df['START'].shift(-1) - pd.DateOffset(days=1))  
    df['END'].iloc[-1] = datetime.fromordinal(end.toordinal())
    return df

def last_month_last_TH():
    from dateutil.relativedelta import TH, relativedelta
    from datetime import datetime, date, timedelta
    lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    lt = lm + relativedelta(weekday=TH(-1))
    return lt

def next_month_last_TH():
    from dateutil.relativedelta import TH, relativedelta
    from datetime import datetime, date, timedelta
    lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    lt = lm + relativedelta(weekday=TH(4))
    return lt

def next_n_thursdays(nths=10):
    import pandas as pd
    from datetime import date
    return pd.date_range(start=date.today(), periods=nths, freq='W-THU').tolist()
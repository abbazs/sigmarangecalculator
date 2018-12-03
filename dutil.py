from datetime import datetime, date, timedelta
from dateutil import parser
from dateutil.relativedelta import TH, relativedelta
from pandas.core.series import Series

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

def get_previous_month_last_TH(input_date=None):
    if input_date == None:
        lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    elif process_date(input_date) == None:
        lm = datetime.combine(date.today().replace(day=1), datetime.min.time()) - timedelta(days=1)
    else:
        lm = process_date(input_date).replace(day=1) - timedelta(days=1)
    lt = lm + relativedelta(weekday=TH(-1))
    return lt
from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime, date
import pandas as pd
import numpy as np
from pathlib import Path

from log import print_exception
from excel_util import create_excel_chart

FNO = 'fno'
STOCKS = 'stocks'
IINDEX = 'iindex'
FNORTN = 'fnortn'

sigma_cols=['LR1S', 'UR1S', 'LR2S', 'UR2S', 'LR3S', 'UR3S', 'LR4S', 'UR4S', 'LR5S', 'UR5S', 'LR6S', 'UR6S']
sigmar_cols = [f'{x}r' for x in sigma_cols] + sigma_cols

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

class ldb(object):
    def __init__(self):
        try:
            self.db = create_engine('sqlite:///D:/Work/db/bhav.db')
            self.meta_data = MetaData(self.db)
            self.fno_table = Table(FNO, self.meta_data, autoload=True)
            self.index_table = Table(IINDEX, self.meta_data, autoload=True)
            self.stock_table = Table(STOCKS, self.meta_data, autoload=True)
            self.last_DF = None
            self.last_stm = None
            self.losing_streak_counter = 0
            self.cum_losing_streak_counter = 0
            self.options_strike_increment = 0
            self.results = [] 
        except Exception as e:
            print_exception(e)

    def get_sql_query_statement(self, table, symbol, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            meta = MetaData(self.db)
            dts = Table(table, meta, autoload=True)
            stm = select(['*']).where(and_(dts.c.TIMESTAMP >= start_date, dts.c.TIMESTAMP <= end_date, dts.c.SYMBOL == symbol))
            return stm
        except Exception as e:
            print_exception(e)
            return None

    def get_spot_data_between_dates(self, symbol, instrument, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)

            if "IDX" in instrument:
                self.last_stm = select(['*']).where(and_(self.index_table.c.TIMESTAMP >= start_date, self.index_table.c.TIMESTAMP <= end_date, self.index_table.c.SYMBOL == symbol))
            else:
                self.last_stm = select(['*']).where(and_(self.stock_table.c.TIMESTAMP >= start_date, self.stock_table.c.TIMESTAMP <= end_date, self.stock_table.c.SYMBOL == symbol))

            df = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP'])
            df = df.drop([df.columns[0]], axis=1)
            df = df.sort_values(['TIMESTAMP'])
            self.spot_DF = df.reset_index(drop=True)

            if "IDX" in instrument:
                return self.spot_DF[self.spot_DF.columns[0:6]]
            else:
                return self.spot_DF[self.spot_DF.columns[-2::2].append(self.spot_DF.columns[0:6]).drop('SERIES')]
        except Exception as e:
            print_exception(e)
            return None

    def get_spot_data_for_today(self, symbol, instrument):
        return self.get_spot_data_between_dates(symbol, instrument, get_current_date())

    def get_spot_data_for_last_n_days(self, symbol, instrument, n_days=0):
        end_date = get_current_date()
        # Add 5 days to n_days and filter only required number days
        start_date = end_date - timedelta(days=n_days + 5)
        spot_data = self.get_spot_data_between_dates(symbol, instrument, start_date, end_date)
        st = end_date - timedelta(days=n_days)
        df = spot_data.set_index('TIMESTAMP')
        self.spot_data = df.loc[df.index >= st]
        return self.spot_data

    def get_spot_data_of_index_for_last_n_days(self, index_symbol, n_days):
        return self.get_spot_data_for_last_n_days(index_symbol, 'IDX', n_days=n_days)

    def get_fno_data_between_dates(self, symbol, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            self.last_stm = select(['*']).where(and_(self.fno_table.c.TIMESTAMP >= start_date, self.fno_table.c.TIMESTAMP <= end_date, self.fno_table.c.SYMBOL == symbol))
            self.fno_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP', 'EXPIRY_DT'])
            self.fno_DF.sort_values(['OPTION_TYP', 'EXPIRY_DT', 'TIMESTAMP', 'STRIKE_PR'], inplace=True)
            self.fno_DF.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            print_exception(e)
            return False

    def get_fno_data_for_today(self, symbol):
        return self.get_fno_data_between_dates(symbol, get_current_date())

    def get_fno_data_for_last_n_days(self, symbol, n_days=0):
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days + 5)
        return self.get_fno_data_between_dates(symbol, start_date, end_date)

    def get_upcoming_expiry_dates(self, symbol, instrument):
        end_date = get_current_date()
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT >= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(self.fno_table.c.EXPIRY_DT)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF = self.last_DF.sort_values(['EXPIRY_DT'])
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]

    def get_last_n_expiry_dates(self, symbol, instrument, n_expiry):
        end_date = get_current_date()
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT <= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(desc(self.fno_table.c.EXPIRY_DT)).limit(n_expiry + 1)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF.sort_values(['EXPIRY_DT'], inplace=True)
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]
    
    def get_last_n_expiry_with_starting_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['START_DT'] = df['EXPIRY_DT'].shift(1) + pd.Timedelta('1Day')
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]

    def get_last_n_expiry_to_expiry_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['START_DT'] = df['EXPIRY_DT'].shift(1)
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]
    
    def get_252_days_spot(self, symbol, instrument, end_date=None):
        if end_date is None:
            end_date = get_current_date()
        start_date = end_date - timedelta(days=400)
        df = self.get_spot_data_between_dates(symbol, instrument, start_date, end_date)
        df = df.set_index('TIMESTAMP')
        df2 = self.get_spot_data_between_dates(symbol, instrument, end_date, get_current_date())
        df2 = df2.set_index('TIMESTAMP')
        self.spot_data = pd.concat([df.iloc[-253:], df2]).drop_duplicates()
        return self.spot_data

    def nifty_get_last_n_expiry_dates(self, n_expiry):
        self.nifty_expds = self.get_last_n_expiry_to_expiry_dates('NIFTY', 'FUTIDX', n_expiry=n_expiry)
        return self.nifty_expds

    def nifty_upcoming_expiry_dates(self):
        return self.get_upcoming_expiry_dates('NIFTY', 'FUTIDX')

    def get_nifty_spot_data(self, n_days=1000):
        return self.get_spot_data_for_last_n_days('NIFTY', 'IDX', n_days=n_days)

    def get_banknifty_spot_data(self, n_days=1000):
        return self.get_spot_data_for_last_n_days('BANKNIFTY', 'IDX', n_days=n_days)

    def get_5sigma_for_data(self, df, n_days_to_calculate):
        nd=n_days_to_calculate
        if nd <= 0:
            print(f'Number of days to calculate shall be more than 0, given number of days = {nd}')
        if len(df) > 253:
            df=df.iloc[-253:]
            dfc=df.assign(DR=np.log(df['CLOSE']/df['CLOSE'].shift(1))).iloc[-252:]
            stddev = dfc['DR'].std()
            mean = dfc['DR'].mean()
            rng = []
            from collections import namedtuple
            sr = namedtuple('sr', ['SIGMA', 'LR', 'SPOT', 'UR', 'LRD', 'URD', 'LRS', 'URS'])
            for x in range(1, 6):
                s = dfc['CLOSE'].iloc[-1]
                u = np.round(np.exp((mean*1) + (np.sqrt(nd) * stddev * x)) * s, 2)
                l = np.round(np.exp((mean*1) - (np.sqrt(nd) * stddev * x)) * s, 2)
                r = sr(f'{x}_sig', l, s, u, np.round((l-s), 2), np.round((u-s), 2), int(np.round(l - 50, -2)), int(np.round(u + 50, -2)))
                rng.append(r)
            return pd.DataFrame(rng)
        else:
            print('Cannot calculate sigma')

    @staticmethod
    def calculate_stdv(df):
        if len(df) >= 253:
            dfc=df.assign(PCLOSE=df['CLOSE'].shift(1))
            dfc=dfc.assign(DR=np.log(dfc['CLOSE']/dfc['CLOSE'].shift(1)))
            dfd=dfc.assign(STDv=dfc['DR'].rolling(252).std())
            dfk=dfd.assign(AVGd=dfd['DR'].rolling(252).mean())
            dfk=dfk.assign(PSTDv=dfk['STDv'].shift(1))
            dfk=dfk.assign(PAVGd=dfk['AVGd'].shift(1))
            if dfk['PSTDv'].iloc[-1] is np.NaN:
                dfk['PSTDv'] = dfk['SDTv']
                dfk['PAVGd'] = dfk['AVGd']
                print('Calculation is being done on current day, hence there are no previous day values.')
            return dfk
        else:
            print('Minimum 252 trading days required to calculate stdv and mean')
            return None

    @staticmethod
    def calculate_6sigma(dfk, dfe, round_digit):
        dfe=dfe.assign(LR1Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 1)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR1Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 1)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(LR2Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 2)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR2Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 2)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(LR3Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 3)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR3Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 3)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(LR4Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 4)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR4Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 4)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(LR5Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 5)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR5Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 5)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(LR6Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 6)) * dfe['PCLOSE'], 2))
        dfe=dfe.assign(UR6Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 6)) * dfe['PCLOSE'], 2))
        
        dfe=dfe.assign(LR1S=dfe['LR1Sr'].round(round_digit))
        dfe=dfe.assign(UR1S=dfe['UR1Sr'].round(round_digit))
        dfe=dfe.assign(LR2S=dfe['LR2Sr'].round(round_digit))
        dfe=dfe.assign(UR2S=dfe['UR2Sr'].round(round_digit))
        dfe=dfe.assign(LR3S=dfe['LR3Sr'].round(round_digit))
        dfe=dfe.assign(UR3S=dfe['UR3Sr'].round(round_digit))
        dfe=dfe.assign(LR4S=dfe['LR4Sr'].round(round_digit))
        dfe=dfe.assign(UR4S=dfe['UR4Sr'].round(round_digit))
        dfe=dfe.assign(LR5S=dfe['LR5Sr'].round(round_digit))
        dfe=dfe.assign(UR5S=dfe['UR5Sr'].round(round_digit))
        dfe=dfe.assign(LR6S=dfe['LR6Sr'].round(round_digit))
        dfe=dfe.assign(UR6S=dfe['UR6Sr'].round(round_digit))

        sigma_full = dfk.join(dfe[sigmar_cols]).fillna(method='bfill')
        sigma = sigma_full.dropna()
        return sigma_full, sigma

    def get_6sigma_for_ndays_interval(self, dfk, n_days_to_calculate=5):
        nd=n_days_to_calculate
        if nd == 0:
            print(f'Number of days to calculate shall not be 0, given number of days = {nd}')
            return
        if dfk is not None:
            dfe = dfk.iloc[::nd].sort_index()
            dfe = dfe.assign(NUMD=np.abs(nd))
            dfs = self.calculate_6sigma(dfk, dfe)
            return dfs
        else:
            print('Cannot calculate sigma')

    def get_6sigma_for_num_expiry(self, symbol, instrument, num_expiry):
        nex = self.get_last_n_expiry_dates(symbol, instrument, num_expiry)
        df = self.get_252_days_spot(symbol, instrument, nex['EXPIRY_DT'].iloc[0])
        dfk = self.calculate_stdv(self.spot_data)

    @staticmethod
    def nifty_calculate_sigmas_interval_days(n_days):
        ld = ldb()
        ld.get_nifty_spot_data(n_days=1000)
        dfk = ld.calculate_stdv(ld.spot_data)
        ld.get_6sigma_for_ndays_interval(dfk, n_days_to_calculate=n_days)
        ldb.create_excel_chart(ld.sigma, f'nifty_{n_days}days_{datetime.now():%Y-%b-%d_%H-%M-%S}')

    @staticmethod
    def calculate_sigmas_e2e_index(symbol, n_expiry, expiry):
        ld = ldb()
        
        instrument = 'FUTIDX'
        if expiry == 'weekly':
            instrument = 'OPTIDX'
        pex = ld.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        nex = ld.get_upcoming_expiry_dates(symbol, instrument)
        #Take only the first upcoming expiry date, don't take the other expiry dates
        #Will not have enough data to calculate sigma
        exs = pd.concat([pex, nex.head(1)]).drop_duplicates().rename(columns={'EXPIRY_DT':'index'})
        #Make a duplicate column of expiry date, rename the columns, and set one of the columns as index
        exs = exs.assign(EID=exs['index']).set_index('index')
        exs = exs.asfreq(freq='1B').fillna(method='bfill')
        ld.get_252_days_spot(symbol, instrument, exs['EID'].iloc[0])
        df = ldb.calculate_stdv(ld.spot_data).dropna()
        #spot df frame will not have the current month, next month and far month, expiry dates.
        #Following line addes those dates by creating a business date range using the last index
        #of df frame and exs frame
        #Create additional index
        aidx = pd.bdate_range(df.index[-1], exs.index[-1]).drop_duplicates()
        #First element of aidx is not required, it is already available
        df = df.reindex(df.index.append(aidx[1:]))
        dfn = df.join(exs)
        dfm = dfn.assign(NUMD=dfn.groupby('EID')['EID'].transform('count')).fillna(method='ffill').dropna()
        dfs = ldb.calculate_6sigma(dfm, dfm.groupby('EID').first(), round_digit=-2)
        title = f'{symbol}_{expiry}_EXPIRYS_{n_expiry}'
        create_excel_chart(dfs[1], title, f'{title}_{datetime.now():%Y-%b-%d_%H-%M-%S}')
        odd = {}
        odd['PEX'] = pex
        odd['NEX'] = nex
        odd['EXS'] = exs
        odd['SIG'] = dfs
        odd['LDB'] = ld
        return odd
    
    @staticmethod
    def nifty_calculate_sigmas_1st_month(n_expiry):
        return ldb.calculate_sigmas_e2e_index('NIFTY', n_expiry, 'monthly')

    @staticmethod
    def banknifty_calculate_sigmas_1st_month(n_expiry):
        return ldb.calculate_sigmas_e2e_index('BANKNIFTY', n_expiry, 'monthly')

    @staticmethod
    def banknifty_calculate_sigmas_1st_week(n_expiry):
        return ldb.calculate_sigmas_e2e_index('BANKNIFTY', n_expiry, 'weekly')

    @staticmethod
    def test1():
        ld = ldb()
        df=ld.get_252_days_spot('NIFTY', 'FUTIDX', datetime(2018, 10, 1))
        dfk = ld.calculate_stdv(ld.spot_data)
        dfs = ld.get_6sigma_for_ndays_interval(dfk, n_days_to_calculate=4)
        return dfs

if __name__ == '__main__':
    dbo = ldb()


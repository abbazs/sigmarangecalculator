from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime, date
from dateutil import parser
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from log import print_exception
from excel_util import create_excel_chart, create_work_sheet_chart

FNO = 'fno'
STOCKS = 'stocks'
IINDEX = 'iindex'
FNORTN = 'fnortn'

sigma_cols=['LR1S', 'UR1S', 'LR2S', 'UR2S', 'LR3S', 'UR3S', 'LR4S', 'UR4S', 'LR5S', 'UR5S', 'LR6S', 'UR6S']
sigmar_cols = [f'{x}r' for x in sigma_cols] + sigma_cols

def process_date(input_date):
    if isinstance(input_date, datetime):
        return input_date
    elif isinstance(input_date, str):
        return parser.parse(input_date)
    else:
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

class sigmas(object):
    def __init__(self, symbol, instrument, round_by=100):
        try:
            self.symbol = symbol.upper()
            self.instrument = instrument.upper()
            self.round_by = round_by
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
            self.sigmadf = None
        except Exception as e:
            print_exception(e)

    def get_sql_query_statement(self, table, start_date, end_date=None):
        try:
            symbol = self.symbol
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            meta = MetaData(self.db)
            dts = Table(table, meta, autoload=True)
            stm = select(['*']).where(and_(dts.c.TIMESTAMP >= start_date, dts.c.TIMESTAMP <= end_date, dts.c.SYMBOL == symbol))
            return stm
        except Exception as e:
            print_exception(e)
            return None

    def get_spot_data_between_dates(self, start_date, end_date=None):
        try:
            symbol = self.symbol
            instrument = self.instrument
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

    def get_spot_data_for_today(self):
        return self.get_spot_data_between_dates(get_current_date())

    def get_spot_data_for_last_n_days(self, n_days=0):
        end_date = get_current_date()
        # Add 5 days to n_days and filter only required number days
        start_date = end_date - timedelta(days=n_days + 5)
        spot_data = self.get_spot_data_between_dates(start_date, end_date)
        st = end_date - timedelta(days=n_days)
        df = spot_data.set_index('TIMESTAMP')
        self.spot_data = df.loc[df.index >= st]
        return self.spot_data

    def get_fno_data_between_dates(self, start_date, end_date=None):
        try:
            symbol = self.symbol
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            self.last_stm = select(['*']).where(and_(self.fno_table.c.TIMESTAMP >= start_date, self.fno_table.c.TIMESTAMP <= end_date, self.fno_table.c.SYMBOL == symbol))
            self.fno_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP', 'EXPIRY_DT'])
            self.fno_DF.sort_values(['OPTION_TYP', 'EXPIRY_DT', 'TIMESTAMP', 'STRIKE_PR'], inplace=True)
            self.fno_DF.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            print_exception(e)
            return False

    def get_fno_data_for_today(self):
        return self.get_fno_data_between_dates(start_date=get_current_date())

    def get_fno_data_for_last_n_days(self, n_days=0):
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days + 5)
        return self.get_fno_data_between_dates(start_date, end_date)

    def get_expiry_dates_available_after_given_date(self, st):
        symbol = self.symbol
        instrument = self.instrument
        start_date = st
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.TIMESTAMP >= start_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().order_by(self.fno_table.c.TIMESTAMP)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF = self.last_DF.sort_values(['EXPIRY_DT'])
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]

    def get_expiry_dates_available_on_given_date(self, st):
        symbol = self.symbol
        instrument = self.instrument
        start_date = st - timedelta(days=7)
        end_date = st
        self.last_stm = select([text('TIMESTAMP'), text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.TIMESTAMP >= start_date,\
                 self.fno_table.c.TIMESTAMP <= end_date,\
                 self.fno_table.c.SYMBOL == symbol)).order_by(self.fno_table.c.TIMESTAMP)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT', 'TIMESTAMP'])
        self.last_DF = self.last_DF.sort_values(['EXPIRY_DT'])
        self.last_DF = self.last_DF[self.last_DF['TIMESTAMP'] == self.last_DF['TIMESTAMP'].iloc[-1]]
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]

    def get_upcoming_expiry_dates(self):
        symbol = self.symbol
        instrument = self.instrument
        end_date = get_current_date()
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT >= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(self.fno_table.c.EXPIRY_DT)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF = self.last_DF.sort_values(['EXPIRY_DT'])
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]

    def get_last_n_expiry_dates(self, n_expiry):
        symbol = self.symbol
        instrument = self.instrument
        end_date = get_current_date()
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT <= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(desc(self.fno_table.c.EXPIRY_DT)).limit(n_expiry + 1)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF.sort_values(['EXPIRY_DT'], inplace=True)
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]
    
    def get_last_n_expiry_with_starting_dates(self, n_expiry):
        df = self.get_last_n_expiry_dates(n_expiry)
        df['START_DT'] = df['EXPIRY_DT'].shift(1) + pd.Timedelta('1Day')
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]

    def get_last_n_expiry_to_expiry_dates(self, n_expiry):
        df = self.get_last_n_expiry_dates(n_expiry)
        df['START_DT'] = df['EXPIRY_DT'].shift(1)
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]
    
    def get_252_days_spot(self, end_date=None):
        '''Gets 252 spot data before end date and gets spot data from the remaining days until current day'''
        if end_date is None:
            ed = get_current_date()
        else:
            ed = end_date
        start_date = ed - timedelta(days=400)
        df = self.get_spot_data_between_dates(start_date, ed)
        df = df.set_index('TIMESTAMP')
        if end_date is None:
            self.spot_data = df.iloc[-253:]
        else:
            #Get data after end date till current date and 
            #append to the data
            df2 = self.get_spot_data_between_dates(ed, get_current_date())
            df2 = df2.set_index('TIMESTAMP')
            self.spot_data = pd.concat([df.iloc[-253:], df2]).drop_duplicates()
        return self.spot_data

    def calculate_stdv(self, num_days=252):
        self.stdvdf=None
        try:
            df = self.spot_data
            if len(df) >= 253:
                dfc=df.assign(PCLOSE=df['CLOSE'].shift(1))
                dfc=dfc.assign(DR=np.log(dfc['CLOSE']/dfc['CLOSE'].shift(1)))
                dfd=dfc.assign(STDv=dfc['DR'].rolling(num_days).std())
                dfk=dfd.assign(AVGd=dfd['DR'].rolling(num_days).mean())
                dfk = dfk.dropna()
                dfk=dfk.assign(PSTDv=dfk['STDv'].shift(1))
                dfk=dfk.assign(PAVGd=dfk['AVGd'].shift(1))
                if len(dfk) <= 1:
                    dfk['PCLOSE']= dfk['CLOSE']
                    dfk['PSTDv'] = dfk['STDv']
                    dfk['PAVGd'] = dfk['AVGd']
                    print('Calculation is being done on current day, hence there are no previous day values.')
                self.stdvdf = dfk
            else:
                print('Minimum 252 trading days required to calculate stdv and mean')
        except Exception as e:
            print_exception(e)
        finally:
            return self.stdvdf

    def calculate(self, exs, st):
        self.get_252_days_spot(st)
        df = self.calculate_stdv()
        #spot df frame will not have the current month, next month and far month, expiry dates.
        #Following line addes those dates by creating a business date range using the last index
        #of df frame and exs frame
        #Create additional index
        aidx = pd.bdate_range(df.index[-1], exs.index[-1]).drop_duplicates()
        #First element of aidx is not required, it is already available
        df = df.reindex(df.index.append(aidx[1:]))
        dfn = df.join(exs)
        dfm = dfn.assign(NUMD=dfn.groupby('EID')['EID'].transform('count')).fillna(method='ffill').dropna()
        dfs = self.six_sigma(dfm, dfm.groupby('EID').first()).fillna(method='bfill').dropna()
        self.sigmadf = dfs
        return dfs

    def get_6sigma_for_ndays_interval(self, n_days_to_calculate=5):
        dfk = self.stdvdf
        nd=n_days_to_calculate
        if nd == 0:
            print(f'Number of days to calculate shall not be 0, given number of days = {nd}')
            return
        if dfk is not None:
            dfe = dfk.iloc[::nd].sort_index()
            dfe = dfe.assign(NUMD=np.abs(nd))
            dfk = dfk.join(dfe.assign(EID=dfe.index)['EID'])
            dfs = self.six_sigma(dfk, dfe)
            return dfs
        else:
            print('Cannot calculate sigma')

    def calculate_and_create_sheets(self, nex, df, file_name):
        ewb = pd.ExcelWriter(file_name, engine='openpyxl')
        dfa = df.dropna()
        st = dfa.index[0]
        nex = nex[nex['ED'] >= st]
        for x in nex.iterrows():
            ed = x[1]['ED']
            aidx = pd.bdate_range(dfa.index[-1], ed).drop_duplicates()
            dfi = dfa.assign(EID=ed)
            if len(aidx) > 0:
                dfi = dfi.reindex(dfi.index.append(aidx[1:])).assign(EID=ed)
            else:
                dfi = dfi[dfi.index <= dfi['EID'].iloc[0]]
            dfi = dfi.assign(NUMD=len(dfi))
            dfi = self.six_sigma(dfi, dfi.iloc[0:1])
            dfi[sigma_cols] = dfi[sigma_cols].ffill() 
            create_work_sheet_chart(ewb, dfi, f"{self.symbol} from {st:%d-%b-%Y} to {ed:%d-%b-%Y} {dfi.iloc[0]['NUMD']} trading days", 1)
        ewb.save()

    def six_sigma(self, dfk, dfe):
        round_by = self.round_by
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
        
        dfe=dfe.assign(LR1S=np.round((dfe['LR1Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR1S=np.round((dfe['UR1Sr'] + (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(LR2S=np.round((dfe['LR2Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR2S=np.round((dfe['UR2Sr'] + (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(LR3S=np.round((dfe['LR3Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR3S=np.round((dfe['UR3Sr'] + (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(LR4S=np.round((dfe['LR4Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR4S=np.round((dfe['UR4Sr'] + (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(LR5S=np.round((dfe['LR5Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR5S=np.round((dfe['UR5Sr'] + (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(LR6S=np.round((dfe['LR6Sr'] - (round_by / 2)) / round_by) * round_by)
        dfe=dfe.assign(UR6S=np.round((dfe['UR6Sr'] + (round_by / 2)) / round_by) * round_by)
        self.sigmadf = dfk.join(dfe[sigmar_cols].reindex(dfk.index))
        return self.sigmadf

    @classmethod
    def nifty_interval_days(cls, n_days):
        ld = cls('NIFTY', 'FUTIDX')
        ld.get_spot_data_for_last_n_days(n_days=1000)
        ld.calculate_stdv()
        ld.get_6sigma_for_ndays_interval(n_days_to_calculate=n_days)
        dfs = ld.sigmadf.fillna(method='bfill')
        create_excel_chart(dfs, f'NIFTY_interval_{n_days}', f'nifty_{n_days}days_{datetime.now():%Y-%b-%d_%H-%M-%S}')
        return ld

    @classmethod
    def expiry2expiry(cls, symbol, instrument, n_expiry, round_by):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys'''
        ld = cls(symbol, instrument, round_by)
        pex = ld.get_last_n_expiry_dates(n_expiry)
        uex = ld.get_upcoming_expiry_dates()
        #Take only the first upcoming expiry date, don't take the other expiry dates
        #Will not have enough data to calculate sigma
        nex = pd.concat([pex, uex.head(1)]).drop_duplicates().rename(columns={'EXPIRY_DT':'ST'})
        st = nex.iloc[0]['ST']
        ld.get_252_days_spot(st)
        df = ld.calculate_stdv()
        dfa = df.dropna()
        st = dfa.index[0]
        nex = nex[nex['ST'] >= st]
        nex = nex.assign(ND=nex[nex['ST'] >= st].shift(-1))
        nex = nex.dropna()
        dfis = []
        file_name = f'{symbol}_expiry2expiry_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
        ewb = pd.ExcelWriter(file_name, engine='openpyxl')

        for x in nex.iterrows():
            st = x[1]['ST']
            nd = x[1]['ND']
            aidx = pd.bdate_range(st, nd)
            dfi = dfa.reindex(aidx).assign(EID=nd).dropna()
            dfi = dfi.assign(NUMD=len(dfi))
            dfi = ld.six_sigma(dfi, dfi.iloc[0:1])
            dfi = dfi.ffill()
            create_work_sheet_chart(ewb, dfi, f"{symbol} from {dfi.index[0]:%d-%b-%Y} to {dfi.index[-1]:%d-%b-%Y} {dfi.iloc[0]['NUMD']} trading days", 1)
            dfis.append(dfi)          

        dfix = pd.concat(dfis)
        create_work_sheet_chart(ewb, dfix, f"{symbol} from {nex.iloc[0]['ST']:%d-%b-%Y} to {nex.iloc[-1]['ND']:%d-%b-%Y} {n_expiry} expirys", 0)
        
        ewb.save()
        return ld
    
    @classmethod
    def calculate_sigmas_e2e(cls, symbol, instrument, n_expiry, round_by):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys'''
        ld = cls(symbol, instrument, round_by)
        pex = ld.get_last_n_expiry_dates(n_expiry)
        nex = ld.get_upcoming_expiry_dates()
        #Take only the first upcoming expiry date, don't take the other expiry dates
        #Will not have enough data to calculate sigma
        exs = pd.concat([pex, nex.head(1)]).drop_duplicates().rename(columns={'EXPIRY_DT':'index'})
        #Make a duplicate column of expiry date, rename the columns, and set one of the columns as index
        exs = exs.assign(EID=exs['index']).set_index('index')
        exs = exs.asfreq(freq='1B').fillna(method='bfill')

        odd = ld.calculate(exs, exs['EID'].iloc[0])
        dfs = ld.sigmadf.fillna(method='bfill')
        title = f'{symbol}_EXPIRYS_{n_expiry}'
        create_excel_chart(dfs, title, f'{title}_E2E_{datetime.now():%Y-%b-%d_%H-%M-%S}')
        return ld

    @classmethod
    def from_start_date_to_all_next_expirys(cls, symbol, instrument, start_date, round_by, file_title=None):
        '''
        Caclulates sig sigmas for expiry to expiry for given number expirys in the past and immediate expiry.
        '''
        ld = cls(symbol, instrument, round_by)    
        st = process_date(start_date)
        ld.get_252_days_spot(st)
        df = ld.calculate_stdv()
        nex = ld.get_expiry_dates_available_on_given_date(st).rename(columns={'EXPIRY_DT':'ED'})
        df = df.dropna()
        if file_title is None:
            file_name = f'{symbol}_start_to_all_next_expirys_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
        else:
            file_name = f'{symbol}_{file_title}_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
        ld.calculate_and_create_sheets(nex, df, file_name)
        return ld

    @classmethod
    def from_last_traded_day_till_all_next_expirys(cls, symbol, instrument, round_by):
        '''
        Calculate sigmas from last trading day till the expiry days for the number of expirys asked
        '''
        return sigmas.from_start_date_to_all_next_expirys(symbol, instrument, get_current_date(), round_by, file_title='from_last_traded_day')
    
    @classmethod
    def from_last_expiry_day_till_all_next_expirys(cls, symbol, instrument, round_by):
        '''
        Calculate sigmas from last expiry day till the expiry days for the number of expirys asked
        '''
        pex = sigmas(symbol, instrument).get_last_n_expiry_dates(1)
        return sigmas.from_start_date_to_all_next_expirys(symbol, instrument, pex['EXPIRY_DT'].iloc[-1], round_by, file_title='from_last_expiry_day' )

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f'{sys.argv[1]}, {sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}')
        ld = sigmas.calculate_sigmas_e2e(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        print('Usage nifty, futidx, 1, 50')
        ld = sigmas.calculate_sigmas_e2e('nifty', 'futidx', 1, 50)
    #ld = sigmas.from_last_traded_day_till_expiry('nifty', 'futidx', 50)
    #ld = sigmas.from_start_date_to_all_next_expirys('nifty', 'futidx', '2018-01-01', 50)
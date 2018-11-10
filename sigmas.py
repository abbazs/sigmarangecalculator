from datetime import timedelta, datetime, date
from dateutil import parser
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from hdf5db import hdf5db
from log import print_exception
from excel_util import create_excel_chart, create_work_sheet_chart
import dutil

class sigmas(object):
    sigma_cols =['LR1S', 'UR1S', 'LR2S', 'UR2S', 'LR3S', 'UR3S', 'LR4S', 'UR4S', 'LR5S', 'UR5S', 'LR6S', 'UR6S']
    sigmar_cols = [f'{x}r' for x in sigma_cols] + sigma_cols

    def __init__(self, symbol, instrument, nstdv=252, round_by=100):
        try:
            self.symbol = symbol.upper()
            self.instrument = instrument.upper()
            self.round_by = round_by
            self.NSTDV = nstdv
            self.db = hdf5db(r'D:/Work/hdf5db/indexdb.hdf')
            self.sigmadf = None
        except Exception as e:
            print_exception(e)
 
   
    def get_n_minus_nstdv_plus_uptodate_spot(self, end_date=None):
        '''Gets 252 spot data before end date and gets spot data from the remaining days until current day'''
        if end_date is None:
            ed = dutil.get_current_date()
        else:
            ed = end_date
        start_date = ed - timedelta(days=(self.NSTDV + 5))
        endd = dutil.get_current_date()
        symbol = self.symbol
        if 'nifty' in symbol:
            df = hdf5db.get_index_data_between_dates(symbol, start_date, endd)
        else:
            df = None #Not yet implemented
        self.spot_data = df
        return self.spot_data

    def calculate_stdv(self):
        self.stdvdf=None
        nd = self.NSTDV
        try:
            df = self.spot_data
            if len(df) >= nd:
                dfc=df.assign(PCLOSE=df['CLOSE'].shift(1))
                dfc=dfc.assign(DR=np.log(dfc['CLOSE']/dfc['CLOSE'].shift(1)))
                dfd=dfc.assign(STDv=dfc['DR'].rolling(nd).std())
                dfk=dfd.assign(AVGd=dfd['DR'].rolling(nd).mean())
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
                print(f'Minimum {nd} trading days required to calculate stdv and mean')
        except Exception as e:
            print_exception(e)
        finally:
            return self.stdvdf

    def calculate(self, exs, st):
        self.get_n_spot(st)
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
            dfi[sigmas.sigma_cols] = dfi[sigmas.sigma_cols].ffill() 
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
        self.sigmadf = dfk.join(dfe[sigmas.sigmar_cols].reindex(dfk.index))
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
    def expiry2expiry(cls, symbol, instrument, n_expiry, nstdev, round_by):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys'''
        ld = cls(symbol, instrument, nstdv=nstdev, round_by=round_by)
        pex = ld.get_last_n_expiry_dates(n_expiry)
        uex = ld.get_upcoming_expiry_dates()
        #Take only the first upcoming expiry date, don't take the other expiry dates
        #Will not have enough data to calculate sigma
        nex = pd.concat([pex, uex.head(1)]).drop_duplicates().rename(columns={'EXPIRY_DT':'ST'})
        st = nex.iloc[0]['ST']
        ld.get_n_spot(st)
        df = ld.calculate_stdv()
        dfa = df.dropna()
        st = dfa.index[0]
        nex = nex[nex['ST'] >= st]
        nex = nex.assign(ND=nex[nex['ST'] >= st].shift(-1))
        nex['ST'] = nex['ST'] + timedelta(days=1)
        nex = nex.dropna()
        dfis = []
        file_name = f'{symbol}_expiry2expiry_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
        ewb = pd.ExcelWriter(file_name, engine='openpyxl')

        for x in nex.iterrows():
            st = x[1]['ST']
            nd = x[1]['ND']
            aidx = pd.bdate_range(st, nd)
            dfi = dfa.reindex(aidx).assign(EID=nd)
            dfi = dfi.assign(NUMD=len(dfi))
            dfi = ld.six_sigma(dfi, dfi.iloc[0:1])
            dfi[sigmas.sigma_cols] = dfi[sigmas.sigma_cols].ffill() 
            create_work_sheet_chart(ewb, dfi, f"{symbol} from {dfi.index[0]:%d-%b-%Y} to {dfi.index[-1]:%d-%b-%Y} {dfi.iloc[0]['NUMD']} trading days", 1)
            dfis.append(dfi)          

        dfix = pd.concat(dfis)
        create_work_sheet_chart(ewb, dfix, f"{symbol} from {nex.iloc[0]['ST']:%d-%b-%Y} to {nex.iloc[-1]['ND']:%d-%b-%Y} {n_expiry} expirys", 0)
        
        ewb.save()
        return ld

    @classmethod
    def from_date_to_all_next_expirys(cls, symbol, instrument, from_date, round_by, ndays_sdtv=252, file_title=None):
        '''
        Caclulates sig sigmas for expiry to expiry for given number expirys in the past and immediate expiry.
        '''
        ld = cls(symbol, instrument, nstdv=ndays_sdtv, round_by=round_by)    
        st = dutil.process_date(from_date)
        ld.get_n_spot(st)
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
    def from_last_traded_day_till_all_next_expirys(cls, symbol, instrument, round_by, ndays_sdtv=252):
        '''
        Calculate sigmas from last trading day till the expiry days for the number of expirys asked
        '''
        return sigmas.from_date_to_all_next_expirys(symbol, instrument, get_current_date(), round_by, ndays_sdtv=252, file_title='from_last_traded_day')
    
    @classmethod
    def from_last_expiry_day_till_all_next_expirys(cls, symbol, instrument, round_by, ndays_sdtv=252):
        '''
        Calculate sigmas from last expiry day till the expiry days for the number of expirys asked
        '''
        pex = sigmas(symbol, instrument).get_last_n_expiry_dates(1)
        return sigmas.from_date_to_all_next_expirys(symbol, instrument, pex['EXPIRY_DT'].iloc[-1], round_by, ndays_sdtv=252, file_title='from_last_expiry_day')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f'{sys.argv[1]}, {sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}, {sys.argv[5]}')
        ld = sigmas.expiry2expiry(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print('Usage nifty, futidx, 1, 50')
        ld = sigmas.expiry2expiry('nifty', 'futidx', 1, 252, 50)
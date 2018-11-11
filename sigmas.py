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
    sigmal_cols = [f'LR{x}S' for x in range(1, 7)]
    sigmau_cols = [f'UR{x}S' for x in range(1, 7)]
    #Join two list one after another
    sigma_cols = [None] * 12
    sigma_cols[::2] = sigmal_cols
    sigma_cols[1::2] = sigmau_cols
    #Sigma true value cols
    sigmar_cols = [f'{x}r' for x in sigma_cols]
    sigmat_cols = [f'{x}t' for x in sigma_cols]
    #All sigma cols
    sigma_all_cols = sigmar_cols + sigma_cols + sigmat_cols + ['SR']

    ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']

    def __init__(self, symbol, instrument, nstdv=252, round_by=100):
        try:
            self.symbol = symbol.upper()
            self.instrument = instrument.upper()
            self.round_by = round_by
            self.NSTDV = nstdv
            self.db = hdf5db(r'D:/Work/hdf5db/indexdb.hdf', self.symbol, self.instrument)
            self.sigmadf = None
        except Exception as e:
            print_exception(e)
 
   
    def get_n_minus_nstdv_plus_uptodate_spot(self, end_date=None):
        '''Gets 252 spot data before end date and gets spot data from the remaining days until current day'''
        if end_date is None:
            ed = dutil.get_current_date()
        else:
            ed = end_date
        start_date = ed - timedelta(days=(self.NSTDV * 2))
        endd = dutil.get_current_date()
        if 'NIFTY' in self.symbol:
            df = self.db.get_index_data_between_dates(start_date, endd)
        else:
            print(f'Given symbol {self.symbol} is not yet implemented...')
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
        self.get_n_minus_nstdv_plus_uptodate_spot(st)
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
        for i in range(1, 7):
            dfe = dfe.assign(**{f'LR{i}Sr': np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'], 2)})
            dfe = dfe.assign(**{f'UR{i}Sr': np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'], 2)})
            dfe = dfe.assign(**{f'LR{i}S': np.round((dfe[f'LR{i}Sr'] - (round_by / 2)) / round_by) * round_by})
            dfe = dfe.assign(**{f'UR{i}S': np.round((dfe[f'UR{i}Sr'] + (round_by / 2)) / round_by) * round_by})
            dfe = dfe.assign(**{f'LR{i}St': np.where(dfe[f'LR{i}S'] < dfe['CLOSE'], 0, -1)})
            dfe = dfe.assign(**{f'UR{i}St': np.where(dfe[f'UR{i}S'] > dfe['CLOSE'], 0, 1)})

            # dfe[f'LR{i}Sr'] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'], 2)
            # dfe[f'UR{i}Sr'] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'], 2)
            # dfe[f'LR{i}S'] = np.round((dfe[f'LR{i}Sr'] - (round_by / 2)) / round_by) * round_by
            # dfe[f'UR{i}S'] = np.round((dfe[f'UR{i}Sr'] + (round_by / 2)) / round_by) * round_by
            # dfe[f'LR{i}St'] = np.where(dfe[f'LR{i}S'] < dfe['CLOSE'], 0, -1)
            # dfe[f'UR{i}St'] = np.where(dfe[f'UR{i}S'] > dfe['CLOSE'], 0, 1)

        dfe = dfe.assign(SR=dfe[sigmas.sigmat_cols].sum(axis=1))

        # dfe=dfe.assign(LR1Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 1)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR1Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 1)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(LR2Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 2)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR2Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 2)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(LR3Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 3)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR3Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 3)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(LR4Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 4)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR4Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 4)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(LR5Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 5)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR5Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 5)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(LR6Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 6)) * dfe['PCLOSE'], 2))
        # dfe=dfe.assign(UR6Sr=np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * 6)) * dfe['PCLOSE'], 2))
        
        # dfe=dfe.assign(LR1S=np.round((dfe['LR1Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR1S=np.round((dfe['UR1Sr'] + (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(LR2S=np.round((dfe['LR2Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR2S=np.round((dfe['UR2Sr'] + (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(LR3S=np.round((dfe['LR3Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR3S=np.round((dfe['UR3Sr'] + (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(LR4S=np.round((dfe['LR4Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR4S=np.round((dfe['UR4Sr'] + (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(LR5S=np.round((dfe['LR5Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR5S=np.round((dfe['UR5Sr'] + (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(LR6S=np.round((dfe['LR6Sr'] - (round_by / 2)) / round_by) * round_by)
        # dfe=dfe.assign(UR6S=np.round((dfe['UR6Sr'] + (round_by / 2)) / round_by) * round_by)



        dfm = dfk.join(dfe[sigmas.sigma_all_cols].reindex(dfk.index))
        for i in range(1, 7):
            dfm = dfm.assign(**{f'LR{i}St': np.where(dfm[f'LR{i}S'] < dfm['CLOSE'], 0, -1)})
            dfm = dfm.assign(**{f'UR{i}St': np.where(dfm[f'UR{i}S'] > dfm['CLOSE'], 0, 1)})
        
        self.sigmadf = dfm.assign(SR=dfe[sigmas.sigmat_cols].sum(axis=1))
        return self.sigmadf

    @classmethod
    def expiry2expiry(cls, symbol, instrument, n_expiry, nstdev, round_by):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys'''
        ld = cls(symbol, instrument, nstdv=nstdev, round_by=round_by)
        try:
            pex = ld.db.get_past_n_expiry_dates(n_expiry)
            uex = ld.db.get_next_expiry_dates().iloc[0]
            #Take only the first upcoming expiry date, don't take the other expiry dates
            #Will not have enough data to calculate sigma
            nex = pex.append(uex).drop_duplicates().rename(columns={'EXPIRY_DT':'ST'})
            st = nex.iloc[0]['ST']
            ld.get_n_minus_nstdv_plus_uptodate_spot(st)
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
            cd = dutil.get_current_date()
            for x in nex.iterrows():
                st = x[1]['ST']
                nd = x[1]['ND']
                print(f'Processing {st:%d-%b-%Y}')
                dfi = dfa[st:nd]
                if dfi.index[-1] != nd:
                    #Do this only for the future expirys
                    if nd > cd:
                        aaidx = pd.bdate_range(dfi.index[-1], nd, closed='right')
                        dfi = dfi.reindex(dfi.index.append(aaidx))
                dfi = dfi.assign(EID=nd)
                dfi = dfi.assign(NUMD=len(dfi))
                dfi = ld.six_sigma(dfi, dfi.iloc[0:1])
                dfi[sigmas.sigma_all_cols] = dfi[sigmas.sigma_all_cols].ffill()
                try:
                    m = f"{symbol} from {dfi.index[0]:%d-%b-%Y} to {dfi.index[-1]:%d-%b-%Y} {dfi.iloc[0]['NUMD']} trading days" 
                    create_work_sheet_chart(ewb, dfi, m, 1)
                    dfis.append(dfi) 
                except:
                    print(dfi)         

            dfix = pd.concat(dfis)
            for i in range(1, 7):
                dfix = dfix.assign(LRC=np.where(dfix[f'LR{i}S']<dfix['CLOSE'], i * -1, 0))
            
            mm = f"{symbol} from {nex.iloc[0]['ST']:%d-%b-%Y} to {nex.iloc[-1]['ND']:%d-%b-%Y} {n_expiry} expirys"
            create_work_sheet_chart(ewb, dfix, mm, 0)
            ewb.save()
            ld.sigmadf = dfix
            return ld
        except Exception as e:
            print_exception(e)
            return ld

    @classmethod
    def from_date_to_all_next_expirys(cls, symbol, instrument, from_date, round_by, ndays_sdtv=252, file_title=None):
        '''
        Caclulates sig sigmas for expiry to expiry for given number expirys in the past and immediate expiry.
        '''
        ld = cls(symbol, instrument, nstdv=ndays_sdtv, round_by=round_by)    
        st = dutil.process_date(from_date)
        ld.get_n_minus_nstdv_plus_uptodate_spot(st)
        df = ld.calculate_stdv()
        nex = ld.db.get_expiry_dates_on_date(st).rename(columns={'EXPIRY_DT':'ED'})
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
        ld = sigmas.from_date_to_all_next_expirys(symbol, 
                                                    instrument, 
                                                    dutil.get_current_date(), 
                                                    round_by, 
                                                    ndays_sdtv=252, 
                                                    file_title='from_last_traded_day')
        return ld
    
    @classmethod
    def from_last_expiry_day_till_all_next_expirys(cls, symbol, instrument, round_by, ndays_sdtv=252):
        '''
        Calculate sigmas from last expiry day till the expiry days for the number of expirys asked
        '''
        pex = cls(symbol, instrument).db.get_past_n_expiry_dates(1)
        ld = sigmas.from_date_to_all_next_expirys(symbol, 
                                                    instrument, 
                                                    pex['EXPIRY_DT'].iloc[-1], 
                                                    round_by, 
                                                    ndays_sdtv=ndays_sdtv, 
                                                    file_title='from_last_expiry_day')
        return ld

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f'{sys.argv[1]}, {sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}, {sys.argv[5]}')
        ld = sigmas.expiry2expiry(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print('Usage nifty, futidx, 1, 50')
        ld = sigmas.expiry2expiry('nifty', 'futidx', 1, 252, 50)
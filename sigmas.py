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
    sigmar_cols = [f'{x}r' for x in sigma_cols] + sigma_cols
    sigmat_cols = [f'{x}t' for x in sigma_cols]
    psp_cols = [f'PE{x}' for x in range(1, 7)]
    csp_cols = [f'CE{x}' for x in range(1, 7)]
    #All sigma cols
    sigma_all_cols = sigmar_cols + sigmat_cols

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

    def calculate(self, ewb, df, st, nd, cd):
        try:
            dfi = df[st:nd]
            if dfi.index[-1] != nd:
                #Do this only for the future expirys
                if nd > cd:
                    aaidx = pd.bdate_range(dfi.index[-1], nd, closed='right')
                    dfi = dfi.reindex(dfi.index.append(aaidx))
            dfi = dfi.assign(EID=nd)
            dfi = dfi.assign(NUMD=len(dfi))
            dfi = self.six_sigma(dfi, dfi.iloc[0:1])
            dfi[sigmas.sigmar_cols] = dfi[sigmas.sigmar_cols].ffill()
            dfi = self.mark_spot_in_range(dfi)
            dfi = self.get_strike_price(dfi)
            try:
                m = f"{self.symbol} from {dfi.index[0]:%d-%b-%Y} to {dfi.index[-1]:%d-%b-%Y} {dfi.iloc[0]['NUMD']} trading days" 
                create_work_sheet_chart(ewb, dfi, m, 1)
                return dfi
            except:
                print(dfi)
                return None
        except Exception as e:
            print_exception(e)
            print(f'Index data my not have been updated for {st}')
            return None 

    def mark_spot_in_range(self, dfk):
        try:
            dfk = dfk.join(pd.DataFrame(columns=sigmas.sigmat_cols))
            for i in range(1, 7):
                dfk[[f'LR{i}St']] = np.where(dfk[f'LR{i}S'] > dfk['CLOSE'], -1, 0)
                dfk[[f'UR{i}St']] = np.where(dfk[f'UR{i}S'] < dfk['CLOSE'], 1, 0)
            
            lrsc = [x for x in sigmas.sigmat_cols if 'L' in x]
            dfk = dfk.assign(LRC=dfk[lrsc].sum(axis=1))
            
            ursc = [x for x in sigmas.sigmat_cols if 'U' in x]
            dfk = dfk.assign(URC=dfk[ursc].sum(axis=1))

            return dfk
        except Exception as e:
            print_exception(e)

    def get_strike_price(self, dfk):
        try:
            st = dfk.index[0]
            nd = dfk.index[-1]
            expd = dfk['EID'].iloc[0]
            dfsp = self.db.get_all_strike_data(st=st, nd=nd, expd=expd) 
            dfk = dfk.join(pd.DataFrame(columns=sigmas.psp_cols + sigmas.csp_cols))
            
            for x, y in zip(sigmas.sigmal_cols, sigmas.psp_cols):
                dfk[y] = dfsp[((dfsp['STRIKE_PR'] == dfk[x].iloc[0]) & (dfsp['OPTION_TYP'] == 'PE'))]['CLOSE']
            
            for x, y in zip(sigmas.sigmau_cols, sigmas.csp_cols):
                dfk[y] = dfsp[((dfsp['STRIKE_PR'] == dfk[x].iloc[0]) & (dfsp['OPTION_TYP'] == 'CE'))]['CLOSE']
            
            return dfk
        except Exception as e:
            print_exception(e)

    def six_sigma(self, dfk, dfe):
        try:
            round_by = self.round_by
            dfe = dfe.join(pd.DataFrame(columns=sigmas.sigmar_cols))
            for i in range(1, 7):
                dfe[[f'LR{i}Sr']] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'])
                dfe[[f'UR{i}Sr']] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'])
                dfe[[f'LR{i}S']] = np.round((dfe[f'LR{i}Sr'] - (round_by / 2)) / round_by) * round_by
                dfe[[f'UR{i}S']] = np.round((dfe[f'UR{i}Sr'] + (round_by / 2)) / round_by) * round_by

            self.sigmadf  = dfk.join(dfe[sigmas.sigmar_cols].reindex(dfk.index))
            return self.sigmadf
        except Exception as e:
            print_exception(e)

    @classmethod
    def expiry2expiry(cls, symbol, instrument, n_expiry, nstdv, round_by, num_days_to_expiry=None, which_month=1):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys
        which_month = 1 --> Current Expiry, 2 --> Next Expiry, 3 --> Far Expiry
        '''
        ld = cls(symbol, instrument, nstdv=nstdv, round_by=round_by)
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
        
            if which_month >= 1 and which_month <= 3:
                nex = nex.assign(ND=nex[nex['ST'] >= st].shift(-which_month))
            else:
                print(f"Processing month {which_month} is not yet supported")
                return None

            if num_days_to_expiry is None:
                nex['ST'] = nex['ST'] + timedelta(days=1)
            else:
                nex['ST'] = nex['ND'] - timedelta(days=num_days_to_expiry)
            nex = nex.dropna()
            cd = dutil.get_current_date()
            if nex.iloc[-1]['ST'] > cd:
                nex.iloc[-1]['ST'] = cd

            dfis = []
            file_name = f'{symbol}_expiry2expiry_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
            ewb = pd.ExcelWriter(file_name, engine='openpyxl')
            
            for x in nex.iterrows():
                st = x[1]['ST']
                nd = x[1]['ND']
                print(f'Processing {st:%d-%b-%Y}')
                dfis.append(ld.calculate(ewb, dfa, st, nd, cd))

            dfix = pd.concat(dfis)
            mm = f"{symbol} from {nex.iloc[0]['ST']:%d-%b-%Y} to {nex.iloc[-1]['ND']:%d-%b-%Y} {n_expiry} expirys"
            create_work_sheet_chart(ewb, dfix, mm, 0)
            ewb.save()
            ld.sigmadf = dfix
            return ld
        except Exception as e:
            print_exception(e)
            return ld

    @classmethod
    def from_date_to_all_next_expirys(cls, symbol, instrument, from_date, round_by, nstdv=252, file_title=None):
        '''
        Caclulates sig sigmas for expiry to expiry for given number expirys in the past and immediate expiry.
        '''
        ld = cls(symbol, instrument, nstdv=nstdv, round_by=round_by)    
        st = dutil.process_date(from_date)
        ld.get_n_minus_nstdv_plus_uptodate_spot(st)
        df = ld.calculate_stdv()
        nex = ld.db.get_expiry_dates_on_date(st).rename(columns={'EXPIRY_DT':'ED'})
        if file_title is None:
            file_name = f'{symbol}_start_to_all_next_expirys_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'
        else:
            file_name = f'{symbol}_{file_title}_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx'

        ewb = pd.ExcelWriter(file_name, engine='openpyxl')
        cd = dutil.get_current_date()
        st = nex['TIMESTAMP'].iloc[0]
        nex = nex[nex['ED'] >= st]
        for x in nex['ED'].iteritems():
            nd = x[1]
            print(f'Processing {nd:%d-%b-%Y}')
            ld.calculate(ewb, df, st, nd, cd)

        ewb.save()
        return ld

    @classmethod
    def from_last_traded_day_till_all_next_expirys(cls, symbol, instrument, round_by, nstdv=252):
        '''
        Calculate sigmas from last trading day till the expiry days for the number of expirys asked
        '''
        ld = sigmas.from_date_to_all_next_expirys(symbol, 
                                                    instrument, 
                                                    dutil.get_current_date(), 
                                                    round_by, 
                                                    nstdv=nstdv, 
                                                    file_title='from_last_traded_day')
        return ld
    
    @classmethod
    def from_last_expiry_day_till_all_next_expirys(cls, symbol, instrument, round_by, nstdv=252):
        '''
        Calculate sigmas from last expiry day till the expiry days for the number of expirys asked
        '''
        pex = cls(symbol, instrument).db.get_past_n_expiry_dates(1)
        pex = pex['EXPIRY_DT'].iloc[0] + timedelta(days=1)
        ld = sigmas.from_date_to_all_next_expirys(symbol, 
                                                    instrument, 
                                                    pex, 
                                                    round_by, 
                                                    nstdv=nstdv, 
                                                    file_title='from_last_expiry_day')
        return ld

    @classmethod
    def nifty_from_last_expriy(cls):
        return sigmas.from_last_expiry_day_till_all_next_expirys('NIFTY', 'FUTIDX', nstdv=252, round_by=50)
    
    @classmethod
    def nifty_from_last_traded_date(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys('NIFTY', 'FUTIDX', nstdv=252, round_by=50)

    @classmethod
    def nifty_expiry2expriy(cls, n_expiry):
        return sigmas.expiry2expiry('NIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=50, num_days_to_expiry=None)

    @classmethod
    def nifty_expiry2expriy_nd2e(cls, n_expiry, nd2e):
        return sigmas.expiry2expiry('NIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=50, num_days_to_expiry=nd2e)

    @classmethod
    def nifty_e2e_nm(cls, n_expiry):
        return sigmas.expiry2expiry('NIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=50, num_days_to_expiry=None, which_month=2)

    @classmethod
    def nifty_e2e_fm(cls, n_expiry):
        return sigmas.expiry2expiry('NIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=50, num_days_to_expiry=None, which_month=3)    

    @classmethod
    def banknifty_from_last_expriy(cls):
        return sigmas.from_last_expiry_day_till_all_next_expirys('BANKNIFTY', 'FUTIDX', nstdv=252, round_by=100)
    
    @classmethod
    def banknifty_from_last_traded_date(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys('BANKNIFTY', 'FUTIDX', nstdv=252, round_by=100)

    @classmethod
    def banknifty_from_last_traded_date_options(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys('BANKNIFTY', 'OPTIDX', nstdv=25, round_by=100)

    @classmethod
    def banknifty_expiry2expriy(cls, n_expiry):
        return sigmas.expiry2expiry('BANKNIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=100, num_days_to_expiry=None)
    
    @classmethod
    def banknifty_expiry2expriy_options(cls, n_expiry):
        return sigmas.expiry2expiry('BANKNIFTY', 'OPTIDX', n_expiry=n_expiry, nstdv=25, round_by=100, num_days_to_expiry=None)

    @classmethod
    def banknifty_expiry2expriy_nd2e(cls, n_expiry, nd2e):
        return sigmas.expiry2expiry('BANKNIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=100, num_days_to_expiry=nd2e)

    @classmethod
    def banknifty_e2e_nm(cls, n_expiry):
        return sigmas.expiry2expiry('BANKNIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=100, num_days_to_expiry=None, which_month=2)

    @classmethod
    def banknifty_e2e_fm(cls, n_expiry):
        return sigmas.expiry2expiry('BANKNIFTY', 'FUTIDX', n_expiry=n_expiry, nstdv=252, round_by=100, num_days_to_expiry=None, which_month=3)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f'{sys.argv[1]}, {sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}, {sys.argv[5]}')
        ld = sigmas.expiry2expiry(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print('Usage nifty, futidx, 1, 50')
        ld = sigmas.expiry2expiry('nifty', 'futidx', 10, 252, 50)
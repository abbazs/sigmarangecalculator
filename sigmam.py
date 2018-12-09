#%%
from nse_data import nse_data as nsed
from stdvcal import calculate_stdv
from dutil import get_last_month_last_TH
from datetime import timedelta
from log import print_exception
import pandas as pd
import numpy as np
#%%
class sigmam(object):
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

    def __init__(self, symbol, nstdv=252, round_by=100):
        try:
            self.symbol = symbol.upper()
            self.round_by = round_by
            self.NSTDV = nstdv
            self.sigmadf = None
        except Exception as e:
            print_exception(e)

    def get_data(self):
        exp_dt = get_last_month_last_TH()
        st = exp_dt - timedelta(days=400)
        dts = nsed.get_dates(st)
        self.spot_data = nsed.get_index_data(dts)
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
    
    def six_sigma(self, dfk, dfe):
        try:
            round_by = self.round_by
            dfe = dfe.join(pd.DataFrame(columns=sigmam.sigmar_cols))
            for i in range(1, 7):
                dfe[[f'LR{i}Sr']] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) - (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'])
                dfe[[f'UR{i}Sr']] = np.round(np.exp((dfe['PAVGd'] * dfe['NUMD']) + (np.sqrt(dfe['NUMD']) * dfe['PSTDv'] * i)) * dfe['PCLOSE'])
                dfe[[f'LR{i}S']] = np.round((dfe[f'LR{i}Sr'] - (round_by / 2)) / round_by) * round_by
                dfe[[f'UR{i}S']] = np.round((dfe[f'UR{i}Sr'] + (round_by / 2)) / round_by) * round_by

            self.sigmadf  = dfk.join(dfe[sigmam.sigmar_cols].reindex(dfk.index))
            return self.sigmadf
        except Exception as e:
            print_exception(e)
    
    @classmethod
    def e2e(cls, symbol, n_expiry, nstdv, round_by, num_days_to_expiry=None, which_month=1):
        '''calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys
        which_month = 1 --> Current Expiry, 2 --> Next Expiry, 3 --> Far Expiry
        '''
        ld = cls(symbol, nstdv=nstdv, round_by=round_by)
        try:
            pex = ld.get_data()
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
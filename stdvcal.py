import pandas as pd
import numpy as np
from log import print_exception

def calculate_stdv(df, nstdv):
    stdvdf=None
    nd = nstdv
    try:
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
            stdvdf = dfk
        else:
            print(f'Minimum {nd} trading days required to calculate stdv and mean')
    except Exception as e:
        print_exception(e)
    finally:
        return stdvdf
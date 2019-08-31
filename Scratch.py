#%%
from src.sigmas import sigmas
#%%
ld = sigmas.nifty_e2e(150)
#%%
ld.moi_cols_reordered
#%%
pec=ld.sigmadf['MOIPE'].corr(ld.sigmadf['MOIPE_C'], method='pearson')
cec=ld.sigmadf['MOICE'].corr(ld.sigmadf['MOICE_C'], method='pearson')
print(f'PE pearson correlation {pec}')
print(f'CE pearson correlation {cec}')

#%%
pec=ld.sigmadf['MOIPE'].corr(ld.sigmadf['MOIPE_C'], method='spearman')
cec=ld.sigmadf['MOICE'].corr(ld.sigmadf['MOICE_C'], method='spearman')
print(f'PE spearman correlation {pec}')
print(f'CE spearman correlation {cec}')

#%%
pec=ld.sigmadf['MOIPE'].corr(ld.sigmadf['MOIPE_C'], method='kendall')
cec=ld.sigmadf['MOICE'].corr(ld.sigmadf['MOICE_C'], method='kendall')
print(f'PE kendall correlation {pec}')
print(f'CE kendall correlation {cec}')
#%%
df = ld.sigmadf['2011':]
#%%
pec=df['MOIPE'].corr(df['MOIPE_C'], method='pearson')
cec=df['MOICE'].corr(df['MOICE_C'], method='pearson')
print(f'PE pearson correlation {pec}')
print(f'CE pearson correlation {cec}')

#%%
pec=df['MOIPE'].corr(df['MOIPE_C'], method='spearman')
cec=df['MOICE'].corr(df['MOICE_C'], method='spearman')
print(f'PE spearman correlation {pec}')
print(f'CE spearman correlation {cec}')

#%%
pec=df['MOIPE'].corr(df['MOIPE_C'], method='kendall')
cec=df['MOICE'].corr(df['MOICE_C'], method='kendall')
print(f'PE kendall correlation {pec}')
print(f'CE kendall correlation {cec}')
#%%
df.corr()

#%% Num days to expiry
from sigmas import sigmas
import numpy as np
start=4.0
end=13.0
incr = 1.0
ndays = 22
npast = 210
summs = {}
for i in np.arange(start, end, incr):
    print(f"Processing factor {i:.3f}")
    ld = sigmas.nifty_nd2e(npast, ndays, i)
    # ld = sigmas.nifty_e2e(npast, i)
    summs[i] = ld.summaryper
import pandas as pd
dfs = []
for k in summs:
    df = summs[k].reset_index()
    df.index = [k] * len(df)
    dfs.append(df)
dfo = pd.concat(dfs)
print("Done")
dfo.to_excel(f'{ndays}_days_{start:.3f}_to_{end:.3f}_factor_sigma.xlsx')

#%%
from sigmas import sigmas
import numpy as np
start=5
end=30
incr = 1.0
ndays = 22
npast = 210
summs = {}
for i in np.arange(start, end, incr):
    print(f"Processing factor {i:.3f}")
    ld = sigmas.nifty_nd2e(npast, i, fd=5)
    # ld = sigmas.nifty_e2e(npast, i)
    summs[i] = ld.summaryper
import pandas as pd
dfs = []
for k in summs:
    df = summs[k].reset_index()
    df.index = [k] * len(df)
    dfs.append(df)
dfo = pd.concat(dfs)
print("Done")
dfo.to_excel(f'{ndays}_days_{start:.3f}_to_{end:.3f}_factor_sigma.xlsx')

#%% Next month expiry
from sigmas import sigmas
import numpy as np
start=4.0
end=13.0
incr = 1.0
ndays = 22
npast = 210
summs = {}
for i in np.arange(start, end, incr):
    print(f"Processing factor {i:.3f}")
    ld = sigmas.nifty_e2e_nm(npast, i)
    # ld = sigmas.nifty_e2e(npast, i)
    summs[i] = ld.summaryper
import pandas as pd
dfs = []
for k in summs:
    df = summs[k].reset_index()
    df.index = [k] * len(df)
    dfs.append(df)
dfo = pd.concat(dfs)
print("Done")
dfo.to_excel(f'nm_days_{start:.3f}_to_{end:.3f}_factor.xlsx')

#%%
def crank(vals):
    from scipy.stats import percentileofscore
    print(vals)
    vls = pd.Series(vals)
    # out = percentileofscore(vals, vals[-1])
    # return out

import json
with open('telegram.dat', 'r') as f:
    secrets = json.load(f)

from pyrogram import Client
app = Client(
secrets["token"],
secrets["api_id"],
secrets["api_hash"]
)
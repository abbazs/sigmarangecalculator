#%%
from sigmas import sigmas
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
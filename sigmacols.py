import numpy as np
#SIGMA COLS
#lrange = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
#urange = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
lrange = np.arange(1.0, 7.0, 1.0)
urange = np.arange(1.0, 7.0, 1.0)
sigmal_cols = [f'LR{x:0.1f}S' for x in lrange]
sigmau_cols = [f'UR{x:0.1f}S' for x in urange]
#Join two list one after another
sigma_cols = [None] * 12
sigma_cols[::2] = sigmal_cols
sigma_cols[1::2] = sigmau_cols
#Sigma true value cols
sigmalr_cols = [f'{x}r' for x in sigmal_cols]
sigmaur_cols = [f'{x}r' for x in sigmau_cols]
#Sigma true value cols + rounded to strike cols
sigmarr_cols = sigmalr_cols + sigmaur_cols + sigma_cols
#sigma marked to spot cols
sigmat_cols = [f'{x}t' for x in sigma_cols]
psp_cols = [f'PE{x:0.1f}' for x in lrange]
csp_cols = [f'CE{x:0.1f}' for x in urange]
#
sigmam_cols = [f'{x}M' for x in sigma_cols]
sigmamr_cols = [f'{x}Mr' for x in sigma_cols]
#All sigma cols
sigma_all_cols = []
sigma_all_cols.extend(sigma_cols)
sigma_all_cols.extend(sigmalr_cols)
sigma_all_cols.extend(sigmaur_cols)
sigma_all_cols.extend(sigmat_cols)
sigma_all_cols.extend(sigmam_cols)
sigma_all_cols.extend(sigmamr_cols)
#
ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
moi_cols = ['MOISCE', 'MOISPE', 'MOICE', 'MOIPE', 
'MOIDCE', 'MOIDPE', 'MOIASCE', 'MOIASPE', 
'MOIACE', 'MOIAPE', 'MOIADCE', 'MOIADPE', 
'MOIRSCE', 'MOIRSPE', 'MOIRCE', 'MOIRPE', 
'MOIRDCE', 'MOIRDPE']

moi_cols_reordered = ['MOISPE', 'MOISCE', 
'MOIPE', 'MOICE', 'MOIDPE', 'MOIDCE', 
'MOIASPE', 'MOIASCE', 'MOIAPE', 'MOIACE', 
'MOIADPE', 'MOIADCE', 'MOIRSPE', 'MOIRSCE', 
'MOIRPE', 'MOIRCE', 'MOIRDPE', 'MOIRDCE']

summary_cols = ["NUMD", "PSTDv", "PAVGd", "PZ", "PP", "LRC", "URC"]
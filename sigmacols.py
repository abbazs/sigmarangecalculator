import numpy as np
#SIGMA COLS
# lrange = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
# urange = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
lrange = np.arange(1.0, 4.5, 0.5)
urange = np.arange(1.0, 3.25, 0.25)
sindex = np.unique(np.concatenate(([0], lrange, urange)))
rln = len(lrange) + len(urange)
#
sigmal_cols = [f'LR{x:0.2f}S' for x in lrange]
sigmau_cols = [f'UR{x:0.2}S' for x in urange]
#Join two list one after another
sigma_cols = sigmal_cols + sigmau_cols
#Sigma true value cols
sigmalr_cols = [f'{x}r' for x in sigmal_cols]
sigmaur_cols = [f'{x}r' for x in sigmau_cols]
sigmar_cols = sigmalr_cols + sigmaur_cols
#Sigma true value cols + rounded to strike cols
sigmarr_cols = sigmar_cols + sigma_cols
#sigma marked to spot cols
sigmalt_cols = [f'{x}t' for x in sigmal_cols]
sigmaut_cols = [f'{x}t' for x in sigmau_cols]
sigmat_cols = sigmalt_cols + sigmaut_cols
#
psp_cols = [f'PE{x:0.2f}' for x in lrange]
csp_cols = [f'CE{x:0.2f}' for x in urange]
#
sigmaml_cols = [f'{x}M' for x in sigmal_cols]
sigmamu_cols = [f'{x}M' for x in sigmau_cols] 
sigmam_cols = sigmaml_cols + sigmamu_cols
#
sigmalmr_cols = [f'{x}Mr' for x in sigmal_cols]
sigmaumr_cols = [f'{x}Mr' for x in sigmau_cols]
#
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
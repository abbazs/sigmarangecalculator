import numpy as np

# SIGMA COLS
# lrange = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
# urange = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
rangel = np.arange(0.5, 4.25, 0.25)
rangeu = np.arange(0.5, 4.25, 0.25)
sindex = np.unique(np.concatenate(([0], rangel, rangeu)))
rln = len(rangel) + len(rangeu)
#
sigmal_cols = [f"LR{x:0.2f}S" for x in rangel]
sigmau_cols = [f"UR{x:0.2}S" for x in rangeu]
# Sigma true value cols
sigmalr_cols = [f"{x}r" for x in sigmal_cols]
sigmaur_cols = [f"{x}r" for x in sigmau_cols]
sigmar_cols = sigmalr_cols + sigmaur_cols
# Sigma true value cols + rounded to strike cols
sigmarr_cols = sigmar_cols
# sigma marked to spot cols
sigmalt_cols = [f"{x}t" for x in sigmal_cols]
sigmaut_cols = [f"{x}t" for x in sigmau_cols]
sigmat_cols = sigmalt_cols + sigmaut_cols
#
psp_cols = [f"PE{x:0.2f}" for x in rangel]
csp_cols = [f"CE{x:0.2f}" for x in rangeu]
#
sigmalmr_cols = [f"{x}Mr" for x in sigmal_cols]
sigmaumr_cols = [f"{x}Mr" for x in sigmau_cols]
#
sigmamr_cols = sigmalmr_cols + sigmaumr_cols
#
summary_cols = [
    "NUMD",
    "PSTDv",
    "PAVGd",
    "PZ",
    "PP",
    "PR",
    "LRC",
    "URC",
    "PCLOSE",
    "CLOSE",
]
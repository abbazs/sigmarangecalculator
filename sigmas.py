import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser
from scipy.stats import norm, percentileofscore

import dutil
from excel_util import (
    add_style,
    create_excel_chart,
    create_summary_percentage_sheet,
    create_summary_sheet,
    create_work_sheet_chart,
)
from hdf5db import hdf5db
from log import print_exception
from sigmacols import (
    csp_cols,
    psp_cols,
    rangel,
    rangeu,
    rln,
    sigmalmr_cols,
    sigmalr_cols,
    sigmalt_cols,
    sigmamr_cols,
    sigmar_cols,
    sigmarr_cols,
    sigmat_cols,
    sigmaumr_cols,
    sigmaur_cols,
    sigmaut_cols,
    summary_cols,
)


class sigmas(object):
    WKR = np.arange(7, 12 * 8, 7)
    #
    def __init__(self, symbol, instrument, fd=12.6, round_by=100):
        try:
            self.symbol = symbol.upper()
            self.instrument = instrument.upper()
            self.round_by = round_by
            self.db = hdf5db(
                r"D:/Work/GitHub/hdf5db/indexdb.hdf", self.symbol, self.instrument
            )
            self.sigmadf = None
            self.strikedf = None
            self.summarydf = None
            self.summaryper = None  # Summary percentage
            self.module_path = os.path.abspath(__file__)
            self.module_dir = os.path.dirname(self.module_path)
            self.out_path = Path(self.module_dir).joinpath("output")
            self.fixed_days = fd
            self.NPDAYS = (sigmas.WKR[-2] * self.fixed_days) // 1
        except Exception as e:
            print_exception(e)

    def get_n_minus_npdays_plus_uptodate_spot(self, end_date=None):
        """Gets 252 spot data before end date and gets spot data from the remaining days until current day"""
        if end_date is None:
            ed = dutil.get_current_date()
        else:
            ed = end_date
        start_date = ed - timedelta(days=(self.NPDAYS))
        endd = dutil.get_current_date()
        if "NIFTY" in self.symbol:
            df = self.db.get_index_data_between_dates(start_date, endd)
        else:
            print(f"Given symbol {self.symbol} is not yet implemented...")
            df = None  # Not yet implemented
        self.spot_data = df
        # For debugging
        # file_name = f"{self.symbol}_spot_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx"
        # df.to_excel(file_name)
        return self.spot_data

    def create_stdv_avg_table(self):
        try:
            df = self.spot_data
            fd = self.fixed_days
            df = df.assign(DR=np.log(df["CLOSE"] / df["CLOSE"].shift(1)))
            days_df = pd.DataFrame(
                {"DAYS": np.ceil(sigmas.WKR * fd).astype(int)}, index=sigmas.WKR
            )
            # Standard Deviation
            self.stdv_table = (
                days_df["DAYS"].apply(lambda x: df["DR"].rolling(x).std()).T
            )
            # Average or mean
            self.avg_table = (
                days_df["DAYS"].apply(lambda x: df["DR"].rolling(x).mean()).T
            )
            # Z score or Rank (Refer to box and whisker plot)
            self.z_table = (
                days_df["DAYS"]
                .apply(
                    lambda x: (df["DR"] - df["DR"].rolling(x).mean())
                    / df["DR"].rolling(x).std()
                )
                .T
            )
            # Rank Table
            pctrank = lambda x: pd.Series(x).rank(pct=False).iloc[-1]
            #
            def crank(x):
                vals = pd.Series(x)
                if len(vals) > 252:
                    vals = vals[0:252]
                scr = vals[-1]
                out = percentileofscore(vals, scr)
                return out

            #
            self.rank_table = (
                days_df["DAYS"][1:]
                .apply(lambda x: df["DR"].rolling(x).apply(crank, raw=False))
                .T
            )
            self.rank_table[0] = 0
            #
            cd = dutil.get_current_date()
            nd = cd + timedelta(days=100)
            aaidx = pd.bdate_range(self.stdv_table.index[-1], nd, closed="right")
            #
            self.stdv_table = self.stdv_table.reindex(
                self.stdv_table.index.append(aaidx)
            )
            self.avg_table = self.avg_table.reindex(self.avg_table.index.append(aaidx))
            self.z_table = self.z_table.reindex(self.z_table.index.append(aaidx))
            self.p_table = self.z_table.apply(lambda x: norm.cdf(x))
            self.rank_table = self.rank_table.reindex(
                self.rank_table.index.append(aaidx)
            )
            self.stdv_table = self.stdv_table.assign(PCLOSE=df["CLOSE"])
            #
            self.stdv_table = self.stdv_table.shift(1)
            self.avg_table = self.avg_table.shift(1)
            self.z_table = self.z_table.shift(1)
            self.p_table = self.p_table.shift(1)
            self.rank_table = self.rank_table.shift(1)
            #
        except Exception as e:
            print_exception(e)

    @staticmethod
    def get_from_table(table, row):
        """
        row must be having index value defined as name
        """
        day = row[0]
        val = row[1]
        try:
            return table.loc[day][val]
        except Exception as e:
            print_exception(e)
            print(f"Error getting data for {day} - {val}")
            return np.nan

    #
    def calculate(self, ewb, st, nd, cd):
        try:
            print(f"st = {st:%Y%b%d} - nd = {nd:%Y%b%d}")
            dfi = self.spot_data[st:nd]
            # Check if last date in the filtered data is equal to end date
            if dfi.index[-1] != nd:
                # If the last date is greater than current date
                # Do this only for the future expirys
                if nd > cd:
                    aaidx = pd.bdate_range(dfi.index[-1], nd, closed="right")
                    dfi = dfi.reindex(dfi.index.append(aaidx))
            dfi = dfi.join(self.stdv_table[["PCLOSE"]])
            dfi = dfi.assign(DR=np.log(dfi["CLOSE"] / dfi["CLOSE"].shift(1)))
            dfi = dfi.assign(EID=nd)
            npst = np.datetime64(st, "D")
            npnd = np.datetime64(nd, "D")
            npil = dfi.index.to_numpy("datetime64[D]")
            dfi = dfi.assign(NUMD=np.busday_count(npst, npnd, weekmask=np.full(7, 1)) +1)
            dfi = dfi.assign(DTE=np.busday_count(npil, npnd, weekmask=np.full(7, 1)) +1)
            dfi = dfi.assign(
                TDTE=dfi.DTE.apply(lambda x: sigmas.WKR[x <= sigmas.WKR][0])
            )
            ikl = dfi.groupby("TDTE").apply(lambda x: x.index[0])
            dfi = dfi.assign(WDY=ikl[dfi.TDTE].set_axis(dfi.index, axis=0, inplace=False))
            #
            dfi = dfi.assign(
                PSTDv=dfi[["WDY", "TDTE"]].apply(
                    lambda x: sigmas.get_from_table(self.stdv_table, x), axis=1
                )
            )
            #
            dfi = dfi.assign(
                PAVGd=dfi[["WDY", "TDTE"]].apply(
                    lambda x: sigmas.get_from_table(self.avg_table, x), axis=1
                )
            )
            #
            dfi = dfi.assign(
                PZ=dfi[["WDY", "TDTE"]].apply(
                    lambda x: sigmas.get_from_table(self.z_table, x), axis=1
                )
            )
            #
            dfi = dfi.assign(
                PP=dfi[["WDY", "TDTE"]].apply(
                    lambda x: sigmas.get_from_table(self.p_table, x), axis=1
                )
            )
            #
            dfi = dfi.assign(
                PR=dfi[["WDY", "TDTE"]].apply(
                    lambda x: sigmas.get_from_table(self.rank_table, x), axis=1
                )
            )
            #
            if len(dfi) <= 1:
                print(
                    "Calcuation is being done for current day and it is not supported"
                )
                raise Exception("Calculating for current day is not supported")
            #
            dfi = self.six_sigma(dfi, dfi)
            dfi[sigmarr_cols] = dfi[sigmarr_cols].ffill()
            # Populate monthly sigma raw range columns
            dfp = pd.DataFrame(
                np.full((len(dfi), rln), dfi[sigmar_cols].iloc[0]),
                columns=sigmamr_cols,
                index=dfi.index,
            )
            dfi = dfi.join(dfp)
            # Locate spot in range
            dfi = self.mark_spot_in_range(dfi)
            #
            self.sigma_marked_df = dfi
            try:
                m = (
                    f"{self.symbol} "
                    f"from {dfi.index[0]:%d-%b-%Y} "
                    f"to {dfi.index[-1]:%d-%b-%Y} "
                    f"{dfi.iloc[0]['NUMD']} trading days "
                )
                n = f"{dfi['EID'].iloc[0]:%Y-%b-%d}"
                create_work_sheet_chart(ewb, dfi, m, n)
                return dfi
            except Exception as e:
                print_exception(e)
                print("Error in creating worksheet...")
                return None
        except Exception as e:
            print_exception(e)
            print(f"Index data may not have been updated for {st}")
            return None

    def mark_spot_in_range(self, dfk):
        try:
            dfk = dfk.join(pd.DataFrame(columns=sigmat_cols))
            for i, cl in enumerate(zip(rangel, sigmalt_cols)):
                dfk[[cl[1]]] = np.where(dfk[sigmalmr_cols[i]] > dfk["CLOSE"], -cl[0], 0)
            #
            for i, cl in enumerate(zip(rangeu, sigmaut_cols)):
                dfk[[cl[1]]] = np.where(dfk[sigmaumr_cols[i]] < dfk["CLOSE"], cl[0], 0)
            #
            dfk = dfk.assign(LRC=dfk[sigmalt_cols].min(axis=1))
            #
            dfk = dfk.assign(URC=dfk[sigmaut_cols].max(axis=1))
            #
            return dfk
        except Exception as e:
            print_exception(e)

    def six_sigma(self, dfk, dfe):
        try:
            dfe = dfe.join(pd.DataFrame(columns=sigmarr_cols))
            #
            for i, cl in enumerate(zip(rangel, sigmalr_cols)):
                dfe[[cl[1]]] = np.round(
                    np.exp(
                        (dfe["PAVGd"] * dfe["TDTE"])
                        - (np.sqrt(dfe["TDTE"]) * dfe["PSTDv"] * cl[0])
                    )
                    * dfe["PCLOSE"]
                )
            for i, cl in enumerate(zip(rangeu, sigmaur_cols)):
                dfe[[cl[1]]] = np.round(
                    np.exp(
                        (dfe["PAVGd"] * dfe["TDTE"])
                        + (np.sqrt(dfe["TDTE"]) * dfe["PSTDv"] * cl[0])
                    )
                    * dfe["PCLOSE"]
                )
            #
            self.sigmadf = dfk.join(dfe[sigmarr_cols].reindex(dfk.index))
            return self.sigmadf
        except Exception as e:
            print_exception(e)

    @classmethod
    def nifty_weekly_range_experiment(cls, start_date, end_date):
        ld = cls('NIFTY', 'FUTIDX')
        try:
           pex = all_thrursdays_between_dates(start_date, end_date)
        except Exception as e:
            print_exception(e)    


    @classmethod
    def expiry2expiry(
        cls,
        symbol,
        instrument,
        n_expiry,
        fd,
        round_by,
        num_days_to_expiry=None,
        which_month=1,
    ):
        """calculates six sigma range for expiry to expiry for the given number of expirys in the past and immediate expirys
        which_month = 1 --> Current Expiry, 2 --> Next Expiry, 3 --> Far Expiry
        """
        ld = cls(symbol, instrument, fd=fd, round_by=round_by)
        try:
            pex = ld.db.get_past_n_expiry_dates(n_expiry)
            wm = None
            if which_month == 1:
                uex = ld.db.get_next_expiry_dates().iloc[0]
                wm = "cm"
            elif which_month == 2:
                uex = ld.db.get_next_expiry_dates().iloc[0:2]
                wm = "nm"
            elif which_month == 3:
                uex = ld.db.get_next_expiry_dates().iloc[0:3]
                wm = "fm"
            else:
                uex = ld.db.get_next_expiry_dates().iloc[0]
                wm = "us"
            #
            # Take only the first upcoming expiry date, don't take the other expiry dates
            # Will not have enough data to calculate sigma
            nex = pex.append(uex).drop_duplicates().rename(columns={"EXPIRY_DT": "ST"})
            st = nex.iloc[0]["ST"]
            ld.get_n_minus_npdays_plus_uptodate_spot(st)
            ld.create_stdv_avg_table()
            st = ld.spot_data.index[0]
            nex = nex[nex["ST"] >= st]
            #
            if which_month >= 1 and which_month <= 3:
                nex = nex.assign(ND=nex[nex["ST"] >= st].shift(-which_month))
            else:
                print(f"Processing month {which_month} is not yet supported")
                return None
            #
            if num_days_to_expiry is None:
                nex["ST"] = nex["ST"] + timedelta(days=1)
            else:
                nex["ST"] = nex["ND"] - timedelta(days=num_days_to_expiry)
            nex = nex.dropna()
            cd = dutil.get_current_date()
            if nex.iloc[-1]["ST"] > cd:
                nex.iloc[-1]["ST"] = cd
            #
            dfis = []
            if num_days_to_expiry is None:
                file_name = (
                    f"{symbol}_e2e_{wm}_{n_expiry}_"
                    f"{fd:.3f}_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx"
                )
            else:
                file_name = (
                    f"{symbol}_e2e_{wm}_{num_days_to_expiry}_{n_expiry}_"
                    f"{fd:.3f}_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx"
                )
            file_name = Path(ld.out_path).joinpath(file_name)
            ewb = pd.ExcelWriter(file_name, engine="openpyxl")
            add_style(ewb)
            #
            for x in nex.iterrows():
                st = x[1]["ST"]
                nd = x[1]["ND"]
                print(f"Processing {st:%d-%b-%Y}")
                dfis.append(ld.calculate(ewb, st, nd, cd))
            #
            dfix = pd.concat(dfis)
            mm = (
                f"{symbol} from {nex.iloc[0]['ST']:%d-%b-%Y} to "
                f"{nex.iloc[-1]['ND']:%d-%b-%Y} {n_expiry} expirys"
            )
            create_work_sheet_chart(ewb, dfix, mm, "AllData")
            dfsummary = pd.pivot_table(
                dfix,
                values=summary_cols,
                index=["EID"],
                aggfunc={
                    "PCLOSE": "first",
                    "CLOSE": "last",
                    "NUMD": "first",
                    "PSTDv": "first",
                    "PAVGd": "first",
                    "PZ": "first",
                    "PP": "first",
                    "PR": "first",
                    "LRC": min,
                    "URC": max,
                },
            )
            dfss = dfsummary[summary_cols]
            dfss = dfss.assign(ER=np.log(dfss["CLOSE"] / dfss["PCLOSE"]))
            create_summary_sheet(ewb, dfss, file_name)
            # Summary % dataframe
            sigma_idx = np.unique(np.concatenate(([0.0], rangel, rangeu)))
            spl = pd.DataFrame(
                {
                    "LRC": [
                        dfsummary[dfsummary["LRC"] <= -x]["LRC"].count()
                        for x in sigma_idx
                    ]
                },
                index=sigma_idx,
            )

            spu = pd.DataFrame(
                {
                    "URC": [
                        dfsummary[dfsummary["URC"] >= x]["URC"].count()
                        for x in sigma_idx
                    ]
                },
                index=sigma_idx,
            )
            sp = spl.join(spu, how="outer")

            sp = sp.assign(LRCP=sp.LRC / sp.LRC[0])
            sp = sp.assign(URCP=sp.URC / sp.URC[0])
            ld.summaryper = sp
            create_summary_percentage_sheet(ewb, sp[["LRC", "LRCP", "URC", "URCP"]])
            ld.summarydf = dfsummary
            # reverse the sheet order
            ewb.book._sheets.reverse()
            ewb.save()
            ld.sigmadf = dfix
            return ld
        except Exception as e:
            print_exception(e)
            return ld

    #
    @classmethod
    def from_date_to_all_next_expirys(
        cls, symbol, instrument, from_date, round_by, fd=12.6, file_title=None
    ):
        """
        Caclulates sig sigmas for expiry to expiry for given number expirys in the past and immediate expiry.
        """
        ld = cls(symbol, instrument, fd=fd, round_by=round_by)
        st = dutil.process_date(from_date)
        ld.get_n_minus_npdays_plus_uptodate_spot(st)
        ld.create_stdv_avg_table()
        nex = ld.db.get_expiry_dates_on_date(st).rename(columns={"EXPIRY_DT": "ED"})
        if file_title is None:
            file_name = f"{symbol}_start_to_all_next_expirys_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx"
        else:
            file_name = f"{symbol}_{file_title}_{datetime.now():%Y-%b-%d_%H-%M-%S}.xlsx"

        file_name = Path(ld.out_path).joinpath(file_name)
        ewb = pd.ExcelWriter(file_name, engine="openpyxl")
        cd = dutil.get_current_date()
        st = nex["TIMESTAMP"].iloc[0]
        nex = nex[nex["ED"] >= st]
        for x in nex["ED"].iteritems():
            nd = x[1]
            print(f"Processing {nd:%d-%b-%Y}")
            ld.calculate(ewb, st, nd, cd)
        #
        ewb.save()
        return ld

    @classmethod
    def from_last_traded_day_till_all_next_expirys(
        cls, symbol, instrument, round_by, fd=12.6
    ):
        """
        Calculate sigmas from last trading day till the expiry days for the number of expirys asked
        """
        ld = sigmas.from_date_to_all_next_expirys(
            symbol,
            instrument,
            dutil.get_current_date(),
            round_by,
            fd=fd,
            file_title="from_last_traded_day",
        )
        return ld

    @classmethod
    def from_last_expiry_day_till_all_next_expirys(
        cls, symbol, instrument, round_by, fd=12.6
    ):
        """
        Calculate sigmas from last expiry day till the expiry days for the number of expirys asked
        """
        pex = cls(symbol, instrument).db.get_past_n_expiry_dates(1)
        pex = pex["EXPIRY_DT"].iloc[0] + timedelta(days=1)
        ld = sigmas.from_date_to_all_next_expirys(
            symbol,
            instrument,
            pex,
            round_by,
            fd=12.6,
            file_title="from_last_expiry_day",
        )
        return ld

    @classmethod
    def nifty_from_last_expriy(cls):
        return sigmas.from_last_expiry_day_till_all_next_expirys(
            "NIFTY", "FUTIDX", fd=12.6, round_by=50
        )

    @classmethod
    def nifty_from_last_traded_date(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys(
            "NIFTY", "FUTIDX", fd=12.6, round_by=50
        )

    @classmethod
    def nifty_e2e(cls, n_expiry, fd=12.6):
        """Nifty expiry to expiry"""
        return sigmas.expiry2expiry(
            "NIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=50,
            num_days_to_expiry=None,
        )

    @classmethod
    def nifty_nd2e(cls, n_expiry, nd2e, fd=12.6):
        """NIFTY FROM LAST NUMBER OF DAYS TO EXPIRY"""
        return sigmas.expiry2expiry(
            "NIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=50,
            num_days_to_expiry=nd2e,
        )

    @classmethod
    def nifty_e2e_nm(cls, n_expiry, fd=12.6):
        """NIFTY EXPIRY 2 EXPIRY NEXT MONTH"""
        return sigmas.expiry2expiry(
            "NIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=50,
            num_days_to_expiry=None,
            which_month=2,
        )

    @classmethod
    def nifty_e2e_fm(cls, n_expiry, fd=12.6):
        """NIFTY EXPIRY 2 EXPIRY FAR MONTH"""
        return sigmas.expiry2expiry(
            "NIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=50,
            num_days_to_expiry=None,
            which_month=3,
        )

    @classmethod
    def banknifty_from_last_expriy(cls):
        return sigmas.from_last_expiry_day_till_all_next_expirys(
            "BANKNIFTY", "FUTIDX", fd=12.6, round_by=100
        )

    @classmethod
    def banknifty_from_last_traded_date(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys(
            "BANKNIFTY", "FUTIDX", fd=12.6, round_by=100
        )

    @classmethod
    def banknifty_from_last_traded_date_options(cls):
        return sigmas.from_last_traded_day_till_all_next_expirys(
            "BANKNIFTY", "OPTIDX", fd=12.6, round_by=100
        )

    @classmethod
    def banknifty_e2e(cls, n_expiry, fd=12.6):
        """Bank Nifty expiry to expiry"""
        return sigmas.expiry2expiry(
            "BANKNIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=100,
            num_days_to_expiry=None,
        )

    @classmethod
    def banknifty_e2e_o(cls, n_expiry, fd=12.6):
        """Bank Nifty expiry to expiry"""
        return sigmas.expiry2expiry(
            "BANKNIFTY",
            "OPTIDX",
            n_expiry=n_expiry,
            fd=fd,
            round_by=100,
            num_days_to_expiry=None,
        )

    @classmethod
    def banknifty_expiry2expriy_nd2e(cls, n_expiry, nd2e):
        return sigmas.expiry2expiry(
            "BANKNIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=12.6,
            round_by=100,
            num_days_to_expiry=nd2e,
        )

    @classmethod
    def banknifty_e2e_nm(cls, n_expiry):
        return sigmas.expiry2expiry(
            "BANKNIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=12.6,
            round_by=100,
            num_days_to_expiry=None,
            which_month=2,
        )

    @classmethod
    def banknifty_e2e_fm(cls, n_expiry):
        return sigmas.expiry2expiry(
            "BANKNIFTY",
            "FUTIDX",
            n_expiry=n_expiry,
            fd=12.6,
            round_by=100,
            num_days_to_expiry=None,
            which_month=3,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(
            f"{sys.argv[1]}, {sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}, {sys.argv[5]}"
        )
        ld = sigmas.expiry2expiry(
            sys.argv[1],
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
        )
    else:
        print("Usage nifty, futidx, 1, 50")
        ld = sigmas.expiry2expiry("nifty", "futidx", 10, 252, 50)

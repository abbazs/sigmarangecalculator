from datetime import timedelta, datetime, date
from dateutil import parser
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from src.log import print_exception
from src import dutil


class hdf5db(object):
    def __init__(self):
        self._symbol = None
        self._instrument = None
        self._path = None
        pass

    @classmethod
    def get_instance(cls):
        return cls()

    @classmethod
    def from_path_symbol_instrument(cls, pth, symbol, instrument):
        db = cls()
        db.path = pth
        db.symbol = symbol
        db.instrument = instrument
        return db

    @classmethod
    def from_path(cls, pth):
        db = cls()
        db.path = pth
        return db

    @property
    def symbol(self):
        """symbol to be processed"""
        if self._symbol is None:
            raise Exception("Symbol is not yet set.")
        else:
            return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value.upper()

    @property
    def instrument(self):
        """instrument to be processed"""
        if self._instrument is None:
            raise Exception("Symbol is not yet set.")
        else:
            return self._instrument

    @instrument.setter
    def instrument(self, value):
        self._instrument = value.upper()

    def set_symbol_instrument(self, symbol, instrument):
        self.symbol = symbol
        self.instrument = instrument

    @property
    def path(self):
        if self._path is None:
            raise Exception("DB path has not been set.")
        else:
            return self._path
    
    @path.setter
    def path(self, value):
        if value is not None:
            if os.path.exists(value):
                self._path = value
            else:
                raise Exception(f"Unable to find given db in path {value}")
        else:
            raise Exception("Given dbpath is none...")

    def get_past_n_expiry_dates(self, n, instrument=None):
        try:
            s = self.symbol
            if instrument is None:
                i = self.instrument
            else:
                i = instrument
            cd = dutil.get_current_date()
            df = pd.read_hdf(
                self.path,
                "fno",
                where="SYMBOL==s and INSTRUMENT==i and EXPIRY_DT<=cd",
                columns=["EXPIRY_DT"],
            ).sort_values("EXPIRY_DT")
            df = df.drop_duplicates().tail(n)
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print_exception(e)

    def get_next_expiry_dates(self):
        try:
            s = self.symbol
            i = self.instrument
            cd = dutil.get_current_date()
            df = pd.read_hdf(
                self.path,
                "fno",
                where="SYMBOL==s and INSTRUMENT==i and EXPIRY_DT>=cd",
                columns=["EXPIRY_DT"],
            ).sort_values("EXPIRY_DT")
            df = df.drop_duplicates()
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print_exception(e)

    def get_expiry_dates_on_date(self, date):
        try:
            s = self.symbol
            i = self.instrument
            ed = dutil.process_date(date)
            st = ed - timedelta(days=5)
            dbp = self.path
            df = pd.read_hdf(
                dbp,
                "fno",
                where="SYMBOL==s and INSTRUMENT==i and TIMESTAMP>=st and TIMESTAMP<=ed",
                columns=["TIMESTAMP", "EXPIRY_DT"],
            ).sort_values("TIMESTAMP")
            df = df[df["TIMESTAMP"] == df["TIMESTAMP"].iloc[-1]]
            df = df[df["TIMESTAMP"] != df["EXPIRY_DT"]]
            df = df.sort_values("EXPIRY_DT")
            df = df.drop_duplicates()
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print(f"Error processing date {date:%Y-%m-%d}")
            print_exception(e)
            return None

    def get_index_data_between_dates(self, st, nd):
        try:
            s = self.symbol
            std = dutil.process_date(st)
            ndd = dutil.process_date(nd)
            dbp = self.path
            df = pd.read_hdf(
                dbp, "idx", where="SYMBOL==s and TIMESTAMP>=std and TIMESTAMP<=ndd"
            ).sort_values("TIMESTAMP")
            df = df.drop_duplicates()
            df = df.set_index("TIMESTAMP")
            return df
        except Exception as e:
            print_exception(e)

    def get_index_data_for_last_n_days(self, n_days):
        try:
            end_date = dutil.get_current_date()
            # Add 5 days to n_days and filter only required number days
            start_date = end_date - timedelta(days=n_days + 5)
            spot_data = self.get_index_data_between_dates(start_date, end_date)
            st = end_date - timedelta(days=n_days)
            df = spot_data.set_index("TIMESTAMP")
            return df.loc[df.index >= st]
        except Exception as e:
            print_exception(e)

    def get_strike_price(self, st, nd, expd, opt, strike):
        try:
            s = self.symbol
            i = "OPTIDX"
            st = dutil.process_date(st)
            nd = dutil.process_date(nd)
            expd = dutil.process_date(expd)
            dbp = self.path
            rule = (
                "SYMBOL==s and "
                "INSTRUMENT==i and "
                "TIMESTAMP>=st and "
                "TIMESTAMP<=nd and "
                "OPTION_TYP==opt and "
                "STRIKE_PR==strike and "
                "EXPIRY_DT==expd"
            )
            df = (
                pd.read_hdf(
                    dbp,
                    "fno",
                    where=rule,
                    columns=[
                        "TIMESTAMP",
                        "CLOSE",
                        "OPEN_INT",
                        "CHG_IN_OI",
                        "STRIKE_PR",
                    ],
                )
                .set_index("TIMESTAMP")
                .sort_index()
            )
            df = df.drop_duplicates()
            return df
        except Exception as e:
            print_exception(e)

    def get_all_strike_data(self, st, nd, expd):
        """
        Gets the strike data between given start and end
        for the given expiry date.
        returns a data frame sorted in ascending order by `TIMESTAMP`
        """
        try:
            s = self.symbol
            i = "OPTIDX"
            st = dutil.process_date(st)
            nd = dutil.process_date(nd)
            expd = dutil.process_date(expd)
            dbp = self.path
            rule = (
                "SYMBOL==s and "
                "INSTRUMENT==i and "
                "TIMESTAMP>=st and "
                "TIMESTAMP<=nd and "
                "EXPIRY_DT==expd"
            )
            df = pd.read_hdf(dbp, "fno", where=rule).sort_values("TIMESTAMP")
            df = df.drop_duplicates()
            return df
        except Exception as e:
            print_exception(e)

    @staticmethod
    def remove_false_expiry(df):
        false_expirys = (df["EXPIRY_DT"] - df["EXPIRY_DT"].shift(1)).dt.days <= 1
        return df[~false_expirys]

    @classmethod
    def get_nifty_instance(cls):
        return cls(
            symbol="nifty", instrument="futidx", pth=r"D:/Work/hdf5db/indexdb.hdf"
        )


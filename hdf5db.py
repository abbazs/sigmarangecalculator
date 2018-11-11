from datetime import timedelta, datetime, date
from dateutil import parser
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from log import print_exception
import dutil

class hdf5db(object):
    def __init__(self, pth, symbol, instrument):
        self.symbol = symbol.upper()
        self.instrument = instrument.upper()
        self.set_db(pth)
    
    def set_db(self, dbpath):
        try:
            if dbpath is not None:
                if os.path.exists(dbpath):
                    self.dbpath = dbpath
                else:
                    print(f'Unable to find given db in path {dbpath}')
            else:
                print('Given dbpath is none...')
        except Exception as e:
            print_exception(e)

    def get_past_n_expiry_dates(self, n):
        try:
            s = self.symbol
            i = self.instrument
            cd = dutil.get_current_date()
            df = pd.read_hdf(self.dbpath, 'fno', 
                            where='SYMBOL==s and INSTRUMENT==i and EXPIRY_DT<=cd',
                            columns=['EXPIRY_DT']).sort_values('EXPIRY_DT')
            df = df.drop_duplicates().tail(n)
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print_exception(e)

    def get_next_expiry_dates(self):
        try:
            s = self.symbol
            i = self.instrument
            cd = dutil.get_current_date()
            df = pd.read_hdf(self.dbpath, 'fno', 
                            where='SYMBOL==s and INSTRUMENT==i and EXPIRY_DT>=cd',
                            columns=['EXPIRY_DT']).sort_values('EXPIRY_DT')
            df = df.drop_duplicates()
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print_exception(e)

    def get_expiry_dates_on_date(self, date):
        try:
            s = self.symbol
            i = self.instrument
            cd = dutil.process_date(date)
            dbp = self.dbpath
            df = pd.read_hdf(dbp, 'fno', 
                            where='SYMBOL==s and INSTRUMENT==i and TIMESTAMP==cd',
                            columns=['EXPIRY_DT']).sort_values('EXPIRY_DT')
            df = df.drop_duplicates()
            return hdf5db.remove_false_expiry(df)
        except Exception as e:
            print_exception(e)

    def get_index_data_between_dates(self, st, nd):
        try:
            s = self.symbol
            std = dutil.process_date(st)
            ndd = dutil.process_date(nd)
            dbp = self.dbpath
            df = pd.read_hdf(dbp, 'idx', 
                            where='SYMBOL==s and TIMESTAMP>=std and TIMESTAMP<=ndd').sort_values('TIMESTAMP')
            df = df.drop_duplicates()
            df = df.set_index('TIMESTAMP')
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
            df = spot_data.set_index('TIMESTAMP')
            return df.loc[df.index >= st]
        except Exception as e:
            print_exception(e) 
    
    @staticmethod
    def remove_false_expiry(df):
        false_expirys = (df['EXPIRY_DT'] - df['EXPIRY_DT'].shift(1)).dt.days <= 1
        return df[~false_expirys]
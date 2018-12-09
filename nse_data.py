import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import zipfile
from io import BytesIO
from io import StringIO
import time
from sqlalchemy import create_engine
import sys, os
from log import print_exception
from bs4 import BeautifulSoup
from dutil import get_last_month_last_TH


class nse_data(object):
    headers = {'Host':'www.nseindia.com'}
    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:56.0) Gecko/20100101 Firefox/56.0'
    headers['Accept'] = '*/*'
    headers['Accept-Language']='en-US,en;q=0.5'
    headers['Accept-Encoding']='gzip, deflate, br'
    headers['Referer']='https://www.nseindia.com/products/content/equities/equities/archieve_eq.htm'
    headers['Connection']='keep-alive'

    idx_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'SHARESTRADED', 'TURNOVER(RS.CR)']
    idx_col_rename = {
    'SHARESTRADED':'VOLUME',
    'TURNOVER(RS.CR)':'TURNOVER'
    }

    @staticmethod
    def get_dates(start):
        end = date.today()
        df = pd.DataFrame(pd.date_range(start=start, end=end, freq='360D'), columns=['START'])
        df = df.assign(END=df['START'].shift(-1) - pd.DateOffset(days=1))  
        df['END'].iloc[-1] = datetime.fromordinal(end.toordinal())
        return df
    
    @staticmethod
    def get_csv_data(urlg, fix_cols):
        pg = requests.get(urlg, headers=nse_data.headers)
        content = pg.content
        if 'No Records' not in content.decode():
            bsf = BeautifulSoup(content, 'html5lib')
            csvc=bsf.find(name='div', attrs={'id':'csvContentDiv'})
            csvs = StringIO(csvc.text.replace(':', '\n'))
            df = pd.read_csv(csvs, error_bad_lines=False)
            cols = {}
            for x in df.columns:
                if 'Date' in x:
                    cols.update({x:'TIMESTAMP'})
                elif 'Prev' in x:
                    cols.update({x:'PCLOSE'})
                else:
                    cols.update({x:x.replace(' ', '').upper()})
            df = df.rename(columns=cols)
            df[fix_cols] = df[fix_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
            df[fix_cols] = df[fix_cols].astype(np.float)
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True)
            return df
        else:
            return None

    @staticmethod
    def get_index_data(dates, index='NIFTY%2050', symbol='NIFTY'):
        try:
            url = 'https://www.nseindia.com/products/dynaContent/equities/indices/historicalindices.jsp?indexType={index}&fromDate={start}&toDate={end}'
            dfs=[]
            for x in dates.iterrows():
                urlg = url.format(index=index, start=x[1][0].strftime('%d-%m-%Y'), end=x[1][1].strftime('%d-%m-%Y'))
                print(urlg)
                dfi = nse_data.get_csv_data(urlg, nse_data.idx_cols)
                if dfi is not None:
                    dfs.append(dfi)
            
            if len(dfs) > 1:
                dfo = pd.concat(dfs)
            elif len(dfs) == 1:
                dfo = dfs[0]
            else:
                dfo = None

            if dfo is not None:
                dfo = dfo.rename(columns=nse_data.idx_col_rename)
                dfo['SYMBOL']=symbol
            
            return dfo
        except Exception as e:
            print(urlg)
            print_exception(e)
            return None

    @staticmethod
    def get_data():
        exp_dt = get_last_month_last_TH()
        st = exp_dt - timedelta(days=400)
        dts = nse_data.get_dates(st)
        dfo = nse_data.get_index_data(dts)
        return dfo



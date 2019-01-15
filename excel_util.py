import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import (
    LineChart,
    StockChart,
    Reference,
    Series,
)
from openpyxl.chart.axis import DateAxis, ChartLines
from openpyxl.chart.updown_bars import UpDownBars
from openpyxl.chart.data_source import NumData, NumVal
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.label import DataLabel
from openpyxl import Workbook
from log import print_exception

def create_line_series(ws, min_col, min_row, max_row, labels, color, legend_loc=0):
    l2 = LineChart()
    l2.add_data(Reference(ws, min_col=min_col, min_row=min_row, max_row=max_row), titles_from_data=True)
    l2.set_categories(labels)
    l2.series[0].graphicalProperties.line.solidFill = color
    s1 = l2.series[0]
    s1.dLbls = DataLabelList()
    #Initialize data label
    dl = DataLabel()
    #Set properties
    dl.showVal = True
    dl.showSerName = True
    dl.idx = legend_loc
    #position t for top
    dl.position = "r"
    #Append data label to data lebels
    s1.dLbls.dLbl.append(dl)
    return l2

def create_work_sheet_chart(ew, df, title, index=1):
    shn = f"{df['EID'].iloc[0]:%Y_%b_%d}"
    if index == 0:
        df.to_excel(excel_writer=ew, sheet_name='SH1')
        ws = ew.book['SH1']
    else:
        df.to_excel(excel_writer=ew, sheet_name=shn)
        ws = ew.book[shn]
        
    dfl = len(df) + 1

    labels = Reference(ws, min_col=1, min_row=2, max_row=dfl)

    ost = df.columns.get_loc('OPEN') + 2
    cnd = df.columns.get_loc('CLOSE') + 2
    
    l1 = StockChart()
    data = Reference(ws, min_col=ost, max_col=cnd, min_row=1, max_row=dfl)
    l1.add_data(data, titles_from_data=True)
    l1.set_categories(labels)
    
    for s in l1.series:
        s.graphicalProperties.line.noFill = True
        
    l1.hiLowLines = ChartLines()
    l1.upDownBars = UpDownBars()
    if title is not None:
        l1.title = title
    else:
        l1.title = shn
    # add dummy cache
    pts = [NumVal(idx=i) for i in range(len(data) - 1)]
    cache = NumData(pt=pts)
    l1.series[-1].val.numRef.numCache = cache

    if dfl <= 6:
        l1.height = 15
        l1.width = 7
    elif dfl >= 6 and dfl <= 25:
        l1.height = 15
        # l1.width = 30
        l1.width = 10 
    else: 
        l1.height = 20
        # l1.width = 40
        l1.width = 10
    #
    # Monthly constant sigma lines
    try:
        sli = df.columns.get_loc('LR1SM') + 2
        sln = sli + 12
        #
        colors = ['ff4554', 'ef6262', 'f17373', 'f38585', 'f49696', 'f6a8a8', 'red', 'blue'] 
        #
        for i, xy in enumerate(zip(range(sli, sln, 2), range(sli + 1, sln, 2))):
            l1 += create_line_series(ws, xy[0], 1, dfl, labels, colors[i], 
            legend_loc=0)
            l1 += create_line_series(ws, xy[1], 1, dfl, labels, colors[i], 
            legend_loc=0)
    except Exception as e:
        print_exception(e)
        print(f'Unable to plot sigmam cols')
    # Moving sigma lines for remaining trading days
    try:
        colors = ['afc1f0', 'bdcbf3', 'cad5f5', 'd7e0f7', 'e4eafa', 'f1f4fc', 'a8e6cf', 'red', 'blue'] 
        #
        sli = df.columns.get_loc('LR1S') + 2
        sln = sli + 12
        # Daily moving sigma lines
        for i, xy in enumerate(zip(range(sli, sln, 2), range(sli + 1, sln, 2))):
            #Change legend_loc value to dfl-2 to get the data lable
            l1 += create_line_series(ws, xy[0], 1, dfl, labels, colors[i], legend_loc=dfl) 
            l1 += create_line_series(ws, xy[1], 1, dfl, labels, colors[i], legend_loc=dfl)
    except Exception as e:
        print_exception(e)
        print(f'Unable to plot sigmam cols')
    #
    mn = df['LR6S'].min() - 100
    mx = df['UR6S'].max() + 100
    l1.x_axis.number_format='yyyymmmdd'
    l1.y_axis.scaling.min = mn
    l1.y_axis.scaling.max = mx
    l1.y_axis.majorUnit = 200
    l1.legend = None
    ws.add_chart(l1, "A2")
    return ws

def create_excel_chart(df, title, name):
    ewb = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')
    create_work_sheet_chart(ewb, df, title, 0)
    for name, g in df.groupby('EID'):
        create_work_sheet_chart(ewb, g, None, 1)
    ewb.save()
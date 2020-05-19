## This project is for MGMTMFE 432-2 Data Analytics and Machine Learning
## Scope: 
## Select 5 pairs of stocks in different industries
## Use WRDS CRSP data of Share Volumn, Daily closed price, Holding Period Return to contruct T-score
## Trade based on T-score
## Use SVM to determine/predict its movement tomorrow
## Short or Long based on the prediction
## Cross Validation on Sample period 2011-2017
## Test on stocks performance in 2019
""" CITATION
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}

https://towardsdatascience.com/intro-to-support-vector-machines-with-a-trading-example-1d4a7997ced6

"""


from bs4 import BeautifulSoup
import datetime
import json
import numpy as np
import pandas as pd
import requests
import time
import warnings
warnings.simplefilter('ignore')

## Install Homebrew and then install TA-lib, for RSI calculation
import talib as ta
from talib import MA_Type

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
# Dow_Page = requests.get('https://finance.yahoo.com/quote/%5EDJI/components?p=%5EDJI')
# Dow_Content = Dow_Page.content

# soup = BeautifulSoup(Dow_Content)

# data = list(soup.findAll("td",{"class":"Py(10px) Ta(start) Pend(10px)"}))

# Ticker_List = []
# for i in data:
#     TempData = str(i)
#     if "title" in TempData:
#         TempData = TempData[TempData.find("title"):]
#         TempData = TempData[TempData.find(">")+1:TempData.find("<")]
#         Ticker_List.append(TempData)
#     else:
#         continue
        
# Start_Date = int(time.mktime((2011,1,1,4,0,0,0,0,0)))
# End_Date = int(time.mktime((2019,12,31,4,0,0,0,0,0)))

# def ScrapeYahoo(data_df,ticker, start, end):
    
#     #Form the URL to be scraped
#     Base_Url = 'https://query1.finance.yahoo.com/v8/finance/chart/'
#     Scrape_Url = Base_Url + ticker + "?period1=" + str(start)+"&period2="+str(end)+"&interval=1d"
    
#     #Get data from page
#     r = requests.get(Scrape_Url)
#     Page_Data = r.json()
    
#     # Compile data into a DataFrame
#     Stock_df = pd.DataFrame()
#     Stock_df['DateTime'] = Page_Data['chart']['result'][0]['timestamp']
#     Stock_df['DateTime'] = Stock_df['DateTime'].apply(lambda x: datetime.datetime.fromtimestamp(x).date().isoformat())
#     Stock_df["Open"] = Page_Data["chart"]["result"][0]["indicators"]["quote"][0]["open"]
#     Stock_df["High"] = Page_Data["chart"]["result"][0]["indicators"]["quote"][0]["high"]
#     Stock_df["Low"] = Page_Data["chart"]["result"][0]["indicators"]["quote"][0]["low"]
#     Stock_df["Close"] = Page_Data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
"""     Stock_df["Volume"] = Page_Data["chart"]["result"][0]["indicators"]["quote"][0]["volume"] """"
#     Stock_df = Stock_df.set_index("DateTime")
    
#     #Add data to a dictionary containing all values
#     data_df[ticker] =  Stock_df

# dict, key: ticker, values: dataframe
Stock_Data = {}

# for i in Ticker_List:
#     ScrapeYahoo(Stock_Data, i, Start_Date, End_Date)
#     #print(i + " done")
#     time.sleep(0.5)



"""
    Factor Building
    Data Cleaning
    
"""

## SMA
## WMA
## MFI
## RSI
## Volumn
for i in Ticker_List:
    
#     Stock_Data[i]['High Shifted']=Stock_Data[i]['High'].shift(-1)
#     Stock_Data[i]['Low Shifted'] = Stock_Data[i]['Low'].shift(-1)
#     Stock_Data[i]['Close Shifted'] = Stock_Data[i]['Close'].shift(-1)
    
#     Stock_Data[i]['Upper BBand'], Stock_Data[i]['Middle BBand'],Stock_Data[i]['Lower BBand']= ta.BBANDS(
#         Stock_Data[i]['Close Shifted'], timeperiod=20,)
    
    Stock_Data[i]['RSI'] = ta.RSI(np.array(Stock_Data[i]['Close Shifted']), timeperiod=14)
    """ 
    Signal Part
    
    Stock_Data[i]['SMA'] = 1
    Stock_Data[i]['WMA'] = 1
    Stock_Data[i]['MFI'] = 1
    
    Stock_Data[i]['MFI'] = 1
    """
#     Stock_Data[i]['Macd'], Stock_Data[i]['Macd Signal'], Stock_Data[i]['Macd Hist'] = ta.MACD(
#         Stock_Data[i]['Close Shifted'], fastperiod=12, slowperiod=26, signalperiod=9)

#     Stock_Data[i]['Momentum'] = ta.MOM(Stock_Data[i]['Close Shifted'],timeperiod=12)
     Stock_Data[i]['Returns'] = np.log(Stock_Data[i]['Open']/Stock_Data[i]['Open'].shift(-1)) 


""" Condition of Trading Signal 
for i in Ticker_List:
    Signal_List = []
    for j in Stock_Data[i]['Returns']:
        
        """"
        ## Condition of Trading Signal 
        if ( True ):
        """"
            Signal_List.append("1")
        else:
            Signal_List.append("0")
"""
        
    Stock_Data[i]['Signal'] = Signal_List
    

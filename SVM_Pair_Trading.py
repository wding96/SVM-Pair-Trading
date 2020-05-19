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

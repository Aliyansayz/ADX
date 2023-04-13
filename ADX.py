import pandas as pd
import numpy as np

""" 
## Example 
bar = yfinance or inclusive bar["High"]  bar["Low"] bar["Close"] 
adx    =   ADX(bar , period = 14 )
"""


def sma (array, period ):

    sma = np.empty_like(array)
    sma = np.full( sma.shape , np.nan)
    # Calculate the EMA for each window of 14 values
    for i in range(period, len(array)+1 ):
          sma[i-1] = np.mean(array[i-period:i] , dtype=np.float16)
    return sma 

def ema (array, period ):

    ema = np.empty_like(array)
    ema = np.full( ema.shape , np.nan)
    ema[0] = np.mean(array[0] , dtype=np.float16)
    alpha = 2 / (period + 1)
    # Calculate the EMA for each window of 14 values
    for i in range(1 , len(array) ):
          ema[i] = np.array( (array[i] * alpha +  ema[i-1]  * (1-alpha) ) , dtype=np.float16 )
    return ema 

def adx_calc(dx , period ):
    adx = np.empty_like(dx)
    adx = np.full( dx.shape , np.nan)
    adx[period-1] = np.mean(dx[:period] , dtype=np.float16)
    # Calculate the EMA for each window of 14 values
    for i in range(period, len(dx)  ):
          adx[i] = np.array( (adx[i-1]*13 + dx[i] )/ period , dtype=np.float16)
    return adx 


def true_range( bar , period  ):
  
  high_low, high_close, low_close  = np.array(bars["High"]-bars["Low"],dtype=np.float16 ) , 
  np.array(abs(bars["High"]-bars["Close"].shift()),dtype=np.float16 ) , 
    np.array(abs(bars["Low"]-bars["Close"].shift() ),dtype=np.float16 )
    
  true_range = np.amax (np.hstack( (high_low, high_close, low_close) ).reshape(-1,3),axis=1 )  
  return true_range 

def  ADX(bar , period ):
  
  true_range  = true_range(bar , multiplier)
  highs , lows =   np.array(abs(bar["High"] - bar["High"].shift(1)), dtype=np.float16)  
  , np.array(abs(bar["Low"].shift(1) - bar["Low"]), dtype=np.float16 )
  condition_pdm = highs > lows
  condition_ndm = lows  > highs 
  pdm = np.where(condition_pdm, highs , 0.0 )
  ndm = np.where(condition_ndm, lows , 0.0 )

  pdm , ndm = sma(pdm , period ) , sma(ndm , period)
  pdm_smoothed = ema(pdm , period)
  ndm_smoothed = ema(ndm , period)
  tr_smoothed  = ema(true_range , period)

  pdi = ( pdm_smoothed / tr_smoothed ) * 100
  ndi = ( ndm_smoothed / tr_smoothed ) * 100

  dx = ( abs(pdi - ndi) ) / ( abs(pdi + ndi) ) * 100
  adx =   adx_calc(dx , period )
  return adx

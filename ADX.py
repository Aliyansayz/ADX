import pandas as pd
import numpy as np

""" 
## Example 
bar = yfinance or inclusive bar["High"]  bar["Low"] bar["Close"] 
adx    =   ADX(bar , period = 14 )
"""

def shift(self, array , place):
    array = np.array(array , dtype= np.float16 )
    array =  array.astype(np.float16)
    shifted = np.roll(array, place)
    shifted[0:place] = np.nan

    return shifted

def sma (array, period ):
    
    sma = np.empty_like(array)
    sma = np.full( sma.shape , np.nan)
    
    for i in range(period, len(array)+1 ):
          sma[i-1] = np.mean(array[i-period:i] , dtype=np.float16)
    return sma 

def calc(dx , period ):
    adx = np.empty_like(dx)
    adx = np.full( dx.shape , np.nan)
    adx[period-1] = np.mean(dx[:period] , dtype=np.float16)
    
    for i in range(period, len(dx)  ):
          adx[i] = np.array( (adx[i-1]*13 + dx[i] )/ period , dtype=np.float16)
    return adx 

def true_range( bar , period  ):
  
    high_low, high_close, low_close  = np.array(bars["High"]-bars["Low"],dtype=np.float16 ) , 
    np.array(abs(bars["High"]-bars["Close"].shift()),dtype=np.float16 ) , 
    np.array(abs(bars["Low"]-bars["Close"].shift() ),dtype=np.float16 )

    true_range = np.amax (np.hstack( (high_low, high_close, low_close) ).reshape(-1,3),axis=1 )
    true_range = np.nan_to_num(true_range , nan=0) 

    return true_range 


def smoothed(self, array, period , alpha = None):
    ema = np.empty_like(array)
    ema = np.full( array.shape , np.nan)
    ema[0] = np.mean(array[0] , dtype=np.float16)
    if alpha == None:
      alpha = 1 / ( period )
    
    for i in range(1 , len(array) ):
          ema[i] = np.array( (array[i] * alpha +  ema[i-1]  * (1-alpha) ) , dtype=np.float16 )
    return ema 


  def calc( array , period):
        adx = np.empty_like(array)
        adx = np.full( array.shape , np.nan)
        adx[period-1] = sum( array[:period] ) / period

        for i in range(period, len(array)) :
              adx[i] = (( adx[i-1]*13 ) + array[i] ) / period 
        return adx 

  def  ADX(bar period ):
        import numpy as np
        true_range  = self.true_range(bar , period )
        high = np.array( bar.High , dtype = np.float16 )
        low = np.array( bar.Low , dtype = np.float16
        highs , lows =   high - shift(high , 1 ) ,  shift(low , 1) - low 

        pdm = np.where(highs > lows  , abs(highs) , 0 )
        ndm = np.where(lows  > highs , abs(lows) , 0  )
            
        atr  = sma(true_range , period)
        pdi = ( smoothed( pdm , period)  / atr ) * 100 
        ndi = ( smoothed( ndm , period) / atr ) * 100
        dx = ( abs(pdi - ndi) ) / ( abs(pdi + ndi) ) * 100 
        dx = np.nan_to_num(dx , nan=0)
        adx = calc( dx , period)

        return adx

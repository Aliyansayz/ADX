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
    shifted[0:place] = 0.0

    return shifted

def sma (array, period ):
    
    sma = np.empty_like(array)
    sma = np.full( sma.shape , np.nan)
    
    for i in range(period, len(array)+1 ):
          sma[i-1] = np.mean(array[i-period:i] , dtype=np.float16)
    return sma 


def true_range( bar , period  ):
  
    high_low, high_close, low_close  = np.array(bars["High"]-bars["Low"],dtype=np.float16 ) , 
    np.array(abs(bars["High"]-bars["Close"].shift()),dtype=np.float16 ) , 
    np.array(abs(bars["Low"]-bars["Close"].shift() ),dtype=np.float16 )

    true_range = np.amax (np.hstack( (high_low, high_close, low_close) ).reshape(-1,3),axis=1 )
    #true_range = np.nan_to_num(true_range , nan=0) 

    true_range[true_range == 0] = 0.0001            
    return true_range 


def smoothed( array, period , alpha = None):
    ema = np.empty_like(array)
    ema = np.full( array.shape , np.nan)
    ema[0] = np.mean(array[0] , dtype=np.float16)
    if alpha == None:
      alpha = 1 / ( period )
    
    for i in range(1 , len(array) ):
          ema[i] = np.array( (array[i] * alpha +  ema[i-1]  * (1-alpha) ) , dtype=np.float16 )
    return ema 


  def  ADX(bar period ):
        import numpy as np
        true_range  = self.true_range(bar , period )
        high = np.array( bar.High , dtype = np.float16 )
        low = np.array( bar.Low , dtype = np.float16
        highs , lows =   high - shift(high , 1 ) ,  shift(low , 1) - low 

        pdm = np.where(highs > lows  , abs(highs) , 0 )
        ndm = np.where(lows  > highs , abs(lows) , 0  )
            
        smoothed_atr  = smoothed(true_range , period)
        smoothed_atr[smoothed_atr == 0] = 0.0001            
                       
        pdi = ( smoothed( pdm , period)  / smoothed_atr ) * 100 
        ndi = ( smoothed( ndm , period) / smoothed_atr ) * 100
        dx = ( abs(pdi - ndi) ) / ( abs(pdi + ndi) ) * 100 
                       
        adx = smoothed( dx , period)

        return adx

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import sys
sys.path.append("/Users/quentinlao/Documents/GitHub/Data-augmentation-time-series/Datasets_analysis/res")
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import time





def detect_periods(timeseries, threshold = 5, verbose = False):

    ### DETECT CANDIDATES PERIODS IN A TIME SERIES

    #Determine the threshold
    permuted_powers = [] # Compute the permuted sequence and record maximum power
    for _ in range(100):
        permuted_timeseries = np.random.permutation(timeseries)
        x,y = periodogram(permuted_timeseries)
        permuted_powers.append(np.max(y))
    power_threshold = np.percentile(permuted_powers, 1) # Calculate the power threshold as the 99th largest value

    x,y = periodogram(timeseries)
    spikes = []
    for i, value in enumerate(y): # Detect significant spikes based on the power threshold
        if value > power_threshold:
            spikes.append(i)
    possible_freq_index = np.array(spikes)
    if not( np.any(possible_freq_index) ) : #si possible_freq_index est vide
        return None
    possible_period = np.array([spike for spike in 1/x[np.array(spikes)] if 2 <= spike and spike <= len(timeseries) / 2]) #trimming

    if verbose : 
        print("Périodes candidates : ", possible_period)



    ### VALIDATE CANDIDATES PERIODS IN A TIME SERIES

    autocorrelation = acf(timeseries, nlags = len(timeseries), fft = False)

    def compute_segment_error(sequence, a, b, verbose = False):
        """
        Compute the error of a linear regression on a segment of a sequence

        Parameters
        ----------
        sequence : array-like
            The sequence to be segmented
        a : int
            The first index of the segment
        b : int 
            The last index of the segment

        Returns
        -------
        error : float
            The error of the linear regression on the segment
        """
        x = np.arange(a, b + 1)
        y = sequence[a : b + 1]
        assert len(x) == len(y)

        slope, intercept = np.polyfit(x, y, deg=1)
        predicted_values = slope * x + intercept
        error = np.sum((y - predicted_values) ** 2)
        return error, slope, intercept


    
    def find_best_split(sequence, verbose = False):
        """
        Find the best split point in a sequence

        Parameters
        ----------  
        sequence : array-like
            The sequence to be segmented

        Returns
        -------
        best_split : int    
            The index of the best split point
        """
        for t in range(1, len(sequence) - 1):
            error1, slope1_tmp, intercept1_tmp = compute_segment_error(sequence, 0, t, verbose = verbose)
            error2, slope2_tmp, intercept2_tmp = compute_segment_error(sequence, t, len(sequence) - 1, verbose = verbose)
            if t == 1 : 
                error = error1 + error2
                slope1 = slope1_tmp
                slope2 = slope2_tmp
                intercept1 = intercept1_tmp
                intercept2 = intercept2_tmp
                best_split = 1
            elif error > error1 + error2:
                error = error1 + error2
                slope1 = slope1_tmp
                slope2 = slope2_tmp
                intercept1 = intercept1_tmp
                intercept2 = intercept2_tmp
                best_split = t

        return best_split, slope1, slope2, intercept1, intercept2
    

    def validate_candidate_periods(acf, candidate_periods):
        validated_periods = []

        for period in candidate_periods:


            #Find the search range
            period_k_plus_1 = 1/( (1/period) + 1/len(acf) )
            period_k_minus_1 = 1/( (1/period) - 1/len(acf) )
            lower_bound_search = int( (period + period_k_plus_1)/2 )  - 2
            upper_bound_search = int( (period + period_k_minus_1)/2 ) + 2

            #Find the best split point in the search range
            best_split, slope1, slope2, intercept1, intercept2 = find_best_split(acf[lower_bound_search : upper_bound_search + 1])
            
            #Check if on a hill
            if slope1 >= slope2 : 
                validated_periods.append(period)

                """
                plt.plot([period,period] , [-0.5,1], color = "blue")
                        
                X1 = np.arange(0,best_split + 1)
                X2 = np.arange(best_split, upper_bound_search - lower_bound_search + 1)
                plt.plot(X1 + lower_bound_search, slope1 * X1 + intercept1, color = "green" )
                plt.plot(X2 + lower_bound_search, slope2 * X2 + intercept2, color = "green")
                """
                
        return validated_periods

    validated_period = validate_candidate_periods(autocorrelation, possible_period)

    if verbose :
        print("\nPériodes validées : ", validated_period)


    ## IDENTIFICATION OF CLOSEST PEAK
    closest_period_list = {}
    def find_closest_peak(acf, period):
        """
        Find the closest peak to a given period in the autocorrelation function

        Parameters
        ----------
        acf : array-like
            The autocorrelation function
        period : int
            The period to be compared

        Returns
        -------
        closest_peak : int
            The index of the closest peak
        """
        left_slope = acf[period] - acf[period - 1]
        right_slope = acf[period + 1] - acf[period]

        if left_slope*right_slope < 0 : #if the slope changes sign ==> period is the peak
            return period
        
        #left_slope and right_slope have the same sign
        elif right_slope > 0 : #if the slope is positive ==> the peak is on the right
            return find_closest_peak(acf, period + 1)
        else : #if the slope is negative ==> the peak is on the left
            return find_closest_peak(acf, period - 1)
    
    right_period = []
    for period in validated_period :
        right_period.append( find_closest_peak(autocorrelation, int(period)) )

    print("Temps de calcul de la période : ", t2 - t1)
    print("Temps de calcul de la validation : ", t3 - t2)
    print("Temps de calcul de la recherche du pic : ", t4 - t3)

    return np.unique(np.array(right_period))




if __name__ == "__main__" :

    file_path = "../../../datasets/FordB/FordB_TRAIN.tsv"
    data = pd.read_csv(file_path, sep='\t', header =None)

    A = detect_periods(np.array( data[data[0] == 1] )[7,1:])

    print(A)
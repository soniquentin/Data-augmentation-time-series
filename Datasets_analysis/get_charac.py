 
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from res.imbalance_degree import imbalance_degree
import csv
from numpy import genfromtxt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf


def calc_bhattacharyya(data, file_name) -> float :
    """
    Calcule la distance de Gaussian Bhattacharyya coefficient (GBC)

    Paramètres
    ----------
    data : array-like, shape=(n_samples, n_features)
        Les données à traiter
    file_name : str
        Le nom du fichier
    """


    def bhattacharyya_distance(class1, class2):
        """
            Compute the Bhattacharyya distance between two classes of time series

            Parameters
            ----------
            class1 : array-like, shape=(n_samples, n_features)
                The first class of time series
            class2 : array-like, shape=(n_samples, n_features)
                The second class of time series

            Returns
            -------
            B : float
                The Bhattacharyya distance between the two classes
        """
        # Compute the PCA of each class over time
        try : 
            pca = PCA(n_components=10)
        except ValueError :
            return np.array([[0]])
        pca.fit(np.concatenate((class1, class2)))
        pc1 = pca.transform(class1)
        pc2 = pca.transform(class2)

        # Compute the mean and covariance of each class over the first 5 principal components

        mean1 = np.mean(pc1, axis=0)
        cov1 = np.cov(pc1.T)
        mean2 = np.mean(pc2, axis=0)
        cov2 = np.cov(pc2.T)

        """
        mean1 = np.mean(pc1, axis=0)
        cov1 = np.diag(np.var(pc1, axis=0))
        mean2 = np.mean(pc2, axis=0)
        cov2 = np.diag(np.var(pc2, axis=0))
        """

        
        # Compute the Bhattacharyya distance
        #On calcule la matrice sigma
        try :
            sigma = np.linalg.inv((cov1+cov2)/2)
            sigma_det = np.abs( np.linalg.det((cov1+cov2)/2))
        except Exception as e :
            #Si la matrice est pas inversible, on ajoute un epsilon
            sigma = np.linalg.inv((cov1+cov2)/2 + 1e-6 * np.eye(cov1.shape[0]))
            sigma_det = np.abs( np.linalg.det((cov1+cov2)/2 + 1e-6 * np.eye(cov1.shape[0])) )

        prod_det = np.abs( np.linalg.det(cov1@cov2) )

        exp_part = np.exp( - 1/8 * (mean2 - mean1).reshape(-1,1).T @ sigma @ (mean2 - mean1).reshape(-1,1)  )
        sqrt_part = np.sqrt( np.sqrt(prod_det) )  / np.sqrt(sigma_det)

        B = exp_part*sqrt_part

        if np.isnan(B[0,0]) : #généralement le cas où exp_part = + inf et sqrt_part = 0 ==> en fait, y'a overflow d'entier ==> exp_part devrait très proche de 0
            return np.array([[0]] , dtype=np.float64)

        # Compute the Bhattacharyya distance
        return B

    original = data.copy()
    #On supprime les lignes de data avec NaN
    data = data.dropna()

    unique_labels = np.array(data[0].unique())
    gbc_sum = np.array([[0]], dtype=np.float64)
    count = 0 #Nombre de calculs de GBC

    for i in range(len(unique_labels)) :
        for j in range(i):
            try : 
                bc = bhattacharyya_distance(data[data[0]==unique_labels[i]].iloc[:,1:].to_numpy(), data[data[0]==unique_labels[j]].iloc[:,1:].to_numpy())
                if bc != np.inf :
                    gbc_sum += bc
                    count += 1
            except ValueError as e :
                #n_sample sous le n_component de ses morts
                return np.nan


    #print(f"GBC ({file_name}) : ", gbc_sum[0,0]/count)
    return gbc_sum[0,0]/count



def calc_variances_normalized(data, file_name) -> (float, float) :
    """
    On normalise toutes les données avant d'appliquer la variance

    Paramètres
    ----------
    data : array-like, shape=(n_samples, n_features)
        Les données à traiter
    file_name : str
        Le nom du fichier

    Returns
    -------
    var : float
        La variance moyenne des variances de chaque label   
    stdev : float   
        La variance moyenne des écarts-types de chaque label
    """

    def aux(data_aux) :     
        new_data = np.array(data)[:,1:]
        upper_quantile = np.nanquantile(new_data, 0.9)
        lower_quantile = np.nanquantile(new_data, 0.1)

        new_data = (data - lower_quantile)/(upper_quantile - lower_quantile)

        return new_data.var().mean()

    #Labels et leur compte :
    stdev_acc = 0

    nb_samples = len(data.index)

    unique_labels = np.sort( data[0].unique() )
    for label in unique_labels :
        count = len( data[data[0] == label].index)
        label_var = aux( data[data[0] == label] )

        stdev_acc += (count/nb_samples)*label_var

    return aux(data), stdev_acc/len(unique_labels)



def calc_length(data,file_name) -> int :
    return (len(data.columns) - 1)



def calc_nb(data,file_name) -> (int,int,dict) :
    #Labels et leur compte :
    unique_labels = np.sort( data[0].unique() )
    label_dict = {}
    for label in unique_labels :
        label_dict[label] = len( data[data[0] == label].index)

    return len(data.index), sum(label_dict.values()) / len(label_dict) , {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}



def calc_IR_ID(data,file_name) -> (float,float) :

    unique_labels = np.sort( data[0].unique() )
    classes_count = np.array( [len( data[data[0] == label].index) for label in unique_labels] )
    im_deg = imbalance_degree(classes_count, "EU")
    if im_deg == 0 : #On divise par 10, peut-être que c'est parce que les nombre sont trop grands
        im_deg = imbalance_degree(classes_count//10, "EU")

    #Imbalance ratio (si possible)
    im_rat = None
    if len(unique_labels) == 2 :
        im_rat = np.min(classes_count)/np.max(classes_count)
        im_deg = None
    
    return im_deg, im_rat




def calc_smoothness_each_label(data,file_name) -> dict :
    """
    Calcule la smoothness pour chaque label

    Paramètres
    ----------
    data : array-like, shape=(n_samples, n_features)
        Les données à traiter
    file_name : str
        Le nom du fichier

    Returns
    -------
    smoothness_dict : dict
        Un dictionnaire contenant la smoothness pour chaque label
    """

    

    unique_labels = np.sort( data[0].unique() )
    label_dict = {}
    for label in unique_labels :
        data_label = np.array( data[data[0] == label] )[:,1:]

        #Normalize min = 0, max = 1 for each timeseries (1 is the max of the timeseries)
        data_label = (data_label - np.min(data_label, axis = 1).reshape(data_label.shape[0],1))/(np.max(data_label, axis = 1).reshape(data_label.shape[0],1) - np.min(data_label, axis = 1).reshape(data_label.shape[0],1) )

        individual_smoothness = np.nanstd(  np.diff(data_label) , axis = 1)
        label_dict[label] = individual_smoothness.mean()

        """
            Note :
                Parfois, individual_smoothness.mean() vaut nan. C'est le cas que pour les datasets dans Missing_value_and_variable_length_datasets_adjusted
        """
    
    return label_dict, np.mean( np.array(list(label_dict.values())) ) , np.std( np.array(list(label_dict.values())) )




def calc_periods_nb(data, file_name) -> dict :

    def detect_periods(timeseries, verbose = False):

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
            return 0
        possible_period = np.array([spike for spike in 1/x[np.array(spikes)] if 2 <= spike and spike <= len(timeseries) / 2]) #trimming

        if verbose : 
            print("Périodes candidates : ", possible_period)



        ### VALIDATE CANDIDATES PERIODS IN A TIME SERIES

        autocorrelation = acf(timeseries, nlags = len(timeseries), fft = False)
        assert len(autocorrelation) == len(timeseries)

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
                if t == 1 or error > error1 + error2: 
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
                lower_bound_search = max( int( (period + period_k_plus_1)/2 )  - 2 , 0 )
                upper_bound_search = min( int( (period + period_k_minus_1)/2 ) + 2 , len(acf) - 1 )

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

            if period == 0 or period == len(acf) - 1: #if the period is on the edge of the acf
                return None

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
            closest_peak = find_closest_peak(autocorrelation, int(period)) 
            if closest_peak is not None :
                right_period.append( closest_peak )

        return len(np.unique(np.array(right_period)))

    data_numpy = np.array(data)[:,1:]
    counts_of_periods = np.apply_along_axis(detect_periods, 1, data_numpy)

    return np.mean(counts_of_periods)



def get_datasets_infos(caract_list):

    infos = []
    
    dataset_folder = "../datasets"
    all_files = []

    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for filename in [f for f in filenames if f.endswith("_TRAIN.tsv")]:
            all_files.append( ( os.path.join(dirpath, filename) ,filename )  )
    

    for file_path, file_name in tqdm(all_files) :
        #NAME AND TYPE
        dict_dataset = {"Name" : file_name}

        summary_data = np.array( pd.read_csv("res/DataSummary.csv" ,sep=',') )
        type_dict = { f'{info[2]}_TRAIN.tsv' :  info[1] for info in summary_data} #Dictionnaire {"Car_TRAIN.tsv" : "Spectro"}
        try : 
            dict_dataset["Type"] = type_dict[file_name]
        except KeyError :
            dict_dataset["Type"] = "Homemade"


        data = pd.read_csv(file_path,sep='\t', header =None)
        #OTHER CHARACTERISTICS
        for function, v in caract_list.items() :
            carac_values = function(data, file_name)

            try :
                if "dict" not in str(type(carac_values)) :
                    carac_values = list(carac_values)
                    v = list(v)
                else :
                    raise TypeError
            except TypeError :
                carac_values = [carac_values]
                v = [v]

            for carac_name, carac_value  in zip(v, carac_values) :
                dict_dataset[carac_name] = carac_value

        #FILEPATH
        dict_dataset["Filepath"] = file_path
        infos.append(dict_dataset)


    return infos



def make_csv(info) :

    with open('infos.csv', 'w') as f:
        
        data = np.array( pd.read_csv("res/DataSummary.csv" ,sep=',') )
        type_dict = { f'{info[2]}_TRAIN.tsv' :  info[1] for info in data}

        writer = csv.writer(f)

        writer.writerow(list(info[0].keys()))

        for info_dataset in info :

            writer.writerow(list(info_dataset.values()))


if __name__ == "__main__" :

    CARAC_DICT = {calc_length : "Length",
                  calc_nb : ("Dataset size", "Avg label size", "Details size label"),
                  calc_variances_normalized : ("Dataset variance", "Intra-class variance"),
                  calc_IR_ID : ("ID", "IR"),
                  calc_smoothness_each_label : ("Smoothness", "Mean smoothness", "Dispersion smoothness"), 
                  calc_bhattacharyya : "Bhattacharyya",
                  calc_periods_nb : "Number of periods"
                  }


    info = get_datasets_infos(CARAC_DICT)

    make_csv(info)
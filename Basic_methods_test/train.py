from tools import *
from get_info import print_title
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from imblearn.metrics import geometric_mean_score
import seaborn as sns
from models import get_model
from scipy.stats import ttest_ind
from tqdm import tqdm
import warnings
from rocket.rocket_functions import apply_kernels
import time
import numpy as np

def analyze_newdata(new_data, method, dataset_name,count_label) :
    """
        new_data : DataFrame
        method : str

        Save the TSNE plot and the synthesized data
    """

    ### PLOT TSNE ###
    plt.figure()

    nb_data = len(new_data)

    new_data["Synthesized"] = ["Orginal" if i < np.sum(count_label) else "Synthesized" for i in range(nb_data)] ##Toutes les données synthétisées ont été concaténées à la fin
    new_data["Size"] = [1 if i < np.sum(count_label) else 0.5 for i in range(nb_data)] #Taille de chaque point
    tsne = TSNE(n_components = 2, perplexity = min(nb_data//20, 40))
    data_transformed = tsne.fit_transform(new_data.drop([0, "Synthesized", "Size"], axis = 1))
    colours = sns.color_palette("hls", len(count_label))
    new_data.rename(columns = {0:'label'}, inplace = True)
    sns.scatterplot(x=data_transformed[:,0], y=data_transformed[:,1], hue = new_data["label"], style = new_data["Synthesized"], size = new_data["Size"], legend='full', palette=colours, alpha = 0.5)
    
    # SAVE SYNTHEZISED DATA
    TSNE_plot_folder = f"tests/{dataset_name}/plot_TSNE"
    if not os.path.exists(TSNE_plot_folder) :
        os.makedirs(TSNE_plot_folder)
    
    plt.savefig(f"{TSNE_plot_folder}/{method}.png", dpi = 200) #Normalement, si on lance depuis Dataset_analysis/, ça devrait sauvegarder dans Dataset_analysis/tests



    ### SAVE FINAL (SYNTHESIZED) DATA ###
    synthesized_data = new_data #synthesized_data = new_data[new_data["Synthesized"] == "Synthesized"]

    synthesized_data.rename(columns = {'label':0}, inplace = True)
    synthesized_data.drop(["Synthesized", "Size"], axis = 1, inplace = True)
    # Créer le dossier des datasets augmentés s'il n'existe pas
    if not os.path.exists("../datasets/Augmented/") :
        os.makedirs("../datasets/Augmented/")
    # Créer le dossier du dataset augmenté s'il n'existe pas
    if not os.path.exists(f"../datasets/Augmented/{dataset_name}") :
        os.makedirs(f"../datasets/Augmented/{dataset_name}")
    # Sauvegarder le dataset augmenté
    synthesized_data.to_csv(f"../datasets/Augmented/{dataset_name}/{dataset_name}_{method}_TRAIN.tsv", sep='\t', index = False,  header =None)


#Train and calculate the score
def train(model, new_data, data_test, **kwargs) :


    mdl, model_name = model
    if model_name == "NN" : #Faut modifier un peu les target
        X, y_temp = np.array(new_data.drop([0], axis = 1), dtype = 'float'), np.array(new_data[0])
        unique_labels = np.sort( new_data[0].unique() )
        y = np.zeros( (len(y_temp) , len(unique_labels) ))
        for i in range(len(y_temp)) :
            ind = np.where(unique_labels == y_temp[i])[0][0]
            y[i,ind] = 1

    elif model_name == "LSTM" : 

        #Même modif que pour NN
        X, y_temp = np.array(new_data.drop([0], axis = 1), dtype = 'float'), np.array(new_data[0])


        unique_labels = np.sort( new_data[0].unique() )
        y = np.zeros( (len(y_temp) , len(unique_labels) ))
        for i in range(len(y_temp)) :
            ind = np.where(unique_labels == y_temp[i])[0][0]
            y[i,ind] = 1

        # On fait des tranches de n_steps parce que LTSM prend des timeseries plus petites. Du coup, ça fait beaucoup plus de timeseries ! 
        n_steps = min(X.shape[1], 100)
        X_new, y_new = [], []
        for i in range(len(X)):
            for j in range(len(X[i]) - n_steps + 1):
                X_new.append(X[i][j:j+n_steps])
                y_new.append(y[i])
        X, y = np.array(X_new), np.array(y_new)
        X = X.reshape((X.shape[0], X.shape[1], 1))

    else :
        X, y = np.array(new_data.drop([0], axis = 1), dtype = 'float'), np.array(new_data[0])
    
    if model_name == "KERNEL" : #Faut transformer X
        mdl, kernels = mdl
        X = apply_kernels(X, kernels)

    print("    --> Fitting model...")
    mdl.fit(X, y, **kwargs)

    print("    --> Scores calculation...")
    X_test, y_test = np.array(data_test.drop([0], axis = 1), dtype = 'float'), np.array(data_test[0])

    #Faut transformer X pour quelques classifiers
    if model_name == "KERNEL" : 
        X_test = apply_kernels(X_test, kernels)
    elif model_name == "LSTM" : 
        X_new = []
        separation = [0] #Pour savoir où sont les séparations entre les timeseries
        i = 0
        for i in range(len(X_test)):
            for j in range(len(X_test[i]) - n_steps + 1):
                X_new.append(X_test[i][j:j+n_steps])
                i += 1
            separation.append(i)
        X_test = np.array(X_new)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_pred = mdl.predict(X_test)

    if model_name == "NN" :
        y_pred = np.array([unique_labels[np.argmax(y)] for y in y_pred], dtype = 'int')
    elif model_name == "LSTM" :
        y_pred_tmp = np.array([unique_labels[np.argmax(y)] for y in y_pred], dtype = 'int')
        y_pred = []
        for i in range(len(separation) - 1):
            #On prend la prédiction qui est le plus apparu dans la prédiction de la timeserie
            mini = min( np.min(y_pred_tmp[separation[i]:separation[i+1]]) , 0 )
            y_pred_tmp[separation[i]:separation[i+1]] -= mini
            y_pred.append( np.argmax(np.bincount(y_pred_tmp[separation[i]:separation[i+1]])) + mini )
        y_pred = np.array(y_pred)

    return { "MCC" : matthews_corrcoef(y_test, y_pred), 
            "F1" : f1_score(y_test, y_pred, average = "weighted"), 
            "G-mean" : geometric_mean_score(y_test, y_pred, average = "weighted"), 
            "Acc" :  accuracy_score(y_test, y_pred)}




def make_score_test(data, data_test, dataset_name, model_name = "RF", nb_iteration = 5):
    """
        model_name : "RF" (Random Forest), "NN" (Simple fully connected layer), "DTW_NEIGBOURS" ()
        nb_iteration : Nombre d'entrainements par transformation (une moyenne, c'est quand même plus fiable)

        scores_matrix : de la forme
                    Index        MCC    F1_score   G-mean   Acc   Model   Transformation   Dataset
                    Default_0     *        *          *      *     RF        Default        Wafer
                    ROS_0         *        *          *      *     RF          ROS          Wafer
    
    """

    scores_matrix = pd.DataFrame(columns=['MCC','F1','G-mean','Acc', 'Model', "Transformation", "Dataset"])


    #Get info on labels and their count
    unique_labels = np.sort( data[0].unique() )
    count_label = np.array( [ len(data[data[0] == label].index) for label in unique_labels ] )
    data_per_class = [data[data[0] == label] for label in unique_labels]

    indice_max = np.argmax(count_label)
    max_label_count, label_max = np.max(count_label), unique_labels[indice_max]

    sampling_strategy = {unique_labels[i] : np.max(count_label) for i in range(len(unique_labels))}
    
    t_i = time.time()
    pbar = tqdm( range(nb_iteration) )
    for i in pbar :

        if time.time() - t_i >= 3*3600 : #Si ça prend trop de temps
            break

        #tqdm write message
        user_msg = f" [{dataset_name}] testing on {model_name} "
        pbar.set_description(f'{user_msg:*^50}')

        print_title(" TRAINING of {} (iteration {}/{}) ".format(model_name, i+1, nb_iteration))

        model, kwargs = get_model(model_name = model_name, data = data)


        print("--> Default")
        scores = train(model, data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Default"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Detault_{}".format(i+1)] = scores

    
        print("--> ROS")
        new_data = data.copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "ROS", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count), dataset_name = dataset_name)
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "ROS"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["ROS_{}".format(i+1)] = scores


        print("--> Jittering")
        new_data = data.copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "Jit", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count), dataset_name = dataset_name)
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Jit"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Jit_{}".format(i+1)] = scores
        #Plot TNSE and save synthetized data
        if i == 0 :
            analyze_newdata(new_data, method = "Jittering", dataset_name = dataset_name ,count_label = count_label)


        print("--> TimeWarping")
        new_data = data.copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "TW", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count), dataset_name = dataset_name)
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "TW"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["TW_{}".format(i+1)] = scores
        #Plot TNSE and save synthetized data
        if i == 0 :
            analyze_newdata(new_data, method = "TimeWarping", dataset_name = dataset_name ,count_label = count_label)



        sampling_strategy = {unique_labels[i] : np.max(count_label) for i in range(len(unique_labels))}

        print("--> Basic Smote")
        new_data = timeseries_smote(data , name_trans = "Basic", sampling_strategy = sampling_strategy, dataset_name = dataset_name)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Basic"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Basic_{}".format(i+1)] = scores
        #Plot TNSE and save synthetized data
        if i == 0 :
            analyze_newdata(new_data, method = "Basic", dataset_name = dataset_name ,count_label = count_label)


        print("--> Basic Adasyn")
        try :
            new_data = timeseries_smote(data , name_trans = "Ada", sampling_strategy = sampling_strategy, dataset_name = dataset_name)
            scores = train(model, new_data, data_test, **kwargs) 
            scores["Model"] = model_name
            scores["Transformation"] = "Ada"
            scores["Dataset"] = dataset_name
            scores_matrix.loc["Ada_{}".format(i+1)] = scores
            #Plot TNSE and save synthetized data
            if i == 0 :
                analyze_newdata(new_data, method = "Ada", dataset_name = dataset_name ,count_label = count_label)
        except Exception as e :
            warnings.warn(f"    /!\/!\/!\ Asadyn failed /!\/!\/!\ : {e}")

        """
        print("--> GAN")
        new_data = gan_augmentation(data, dataset_name, sampling_strategy = sampling_strategy)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "GAN"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["GAN{}".format(i+1)] = scores
        #Plot TNSE and save synthetized data
        if i == 0 :
            analyze_newdata(new_data, method = "GAN", dataset_name = dataset_name ,count_label = count_label)
        """

        print("--> DTW SMOTE")
        new_data = dtw_smote(data, dataset_name, sampling_strategy = sampling_strategy)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "DTW-SMOTE"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["DTW-SMOTE{}".format(i+1)] = scores
        #Plot TNSE and save synthetized data
        if i == 0 :
            analyze_newdata(new_data, method = "DTW-SMOTE", dataset_name = dataset_name ,count_label = count_label)


    #pd.set_option('display.max_rows', None)

    return scores_matrix




def make_final_tab(scores_matrix, save_plot_path = "results"):
    """
        Fonction qui permet de créer le tableau final des résultats
        Entrée :
            - scores_matrix : DataFrame contenant les résultats de toutes les expériences
            - save_plot_path : chemin où sauvegarder les plots

        Sortie :
            - final_tab_mean : DataFrame contenant les moyennes des scores pour chaque transformation
            - final_tab_p_value : DataFrame contenant les p-valeurs pour chaque transformation    
    """

    all_trans = scores_matrix["Transformation"].unique()
    all_metrics = ["MCC", "F1", "G-mean", "Acc"]

    final_tab_mean = pd.DataFrame(columns= all_metrics)
    final_tab_p_value = pd.DataFrame()

    for trans in all_trans :
        sub_score_trans = scores_matrix[scores_matrix["Transformation"] == trans].drop(["Model", "Transformation", "Dataset"], axis = 1)
        final_tab_mean.loc[trans] = sub_score_trans.mean(axis = 0)
        
        #Calcul de la p-valeur
        if trans != "Default" :
            for metric in all_metrics :
                default_values = np.array( scores_matrix[scores_matrix["Transformation"] == "Default"].drop(["Model", "Transformation", "Dataset"], axis = 1)[[metric]])
                default_values = np.squeeze(default_values)

                trans_values = np.array( scores_matrix[scores_matrix["Transformation"] == trans].drop(["Model", "Transformation", "Dataset"], axis = 1)[[metric]])
                trans_values = np.squeeze(trans_values)

                test = ttest_ind(default_values, trans_values, equal_var = False)
                t_stat, pvalue = ttest_ind(default_values, trans_values, equal_var = False)

                final_tab_p_value.loc[trans, metric] = pvalue


    
    for metric in all_metrics :
        model_name = list( scores_matrix["Model"].unique() )[0]
        dataset_name = list( scores_matrix["Dataset"].unique() )[0]
        plt.figure(figsize=(10, 10), dpi=80) 
        sns.violinplot(data=scores_matrix, x=metric, y="Transformation").set(title=metric)
        plt.savefig(save_plot_path + "/{}.png".format(metric))
    
    return final_tab_mean, final_tab_p_value



if __name__ == "__main__" :

    data = pd.read_csv("../datasets/Ham/Ham_TRAIN.tsv",sep='\t', header =None)
    data_test = pd.read_csv("../datasets/Ham/Ham_TEST.tsv",sep='\t', header =None)

    make_score_test(data, data_test, "Ham", model_name = "LSTM", nb_iteration = 5)




    


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


def plot_tsne(new_data, method, dataset_name,count_label, unique_labels) :
    """
        Plot tsne des nouvelles données
    """

    plt.figure()

    nb_data = len(new_data)

    new_data["Synthesized"] = ["Orginal" if i < np.sum(count_label) else "Synthesized" for i in range(nb_data)] ##Toutes les données synthétisées ont été concaténées à la fin
    tsne = TSNE(n_components = 2, perplexity = min(nb_data//20, 40))
    data_transformed = tsne.fit_transform(new_data.drop([0, "Synthesized"], axis = 1))
    colours = sns.color_palette("hls", len(unique_labels))
    new_data.rename(columns = {0:'label'}, inplace = True)
    sns.scatterplot(x=data_transformed[:,0], y=data_transformed[:,1], hue = new_data["label"], style = new_data["Synthesized"],legend='full', palette=colours)
    
    #Save plot
    TSNE_plot_folder = f"tests/{dataset_name}/plot_TSNE"
    if not os.path.exists(TSNE_plot_folder) :
        os.makedirs(TSNE_plot_folder)
    
    plt.savefig(f"{TSNE_plot_folder}/{method}.png", dpi = 200) #Normalement, si on lance depuis Dataset_analysis/, ça devrait sauvegarder dans Dataset_analysis/tests


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
    else :
        X, y = np.array(new_data.drop([0], axis = 1), dtype = 'float'), np.array(new_data[0])
    
    if model_name == "KERNEL" : #Faut transformer X
        mdl, kernels = mdl
        X = apply_kernels(X, kernels)

    print("    --> Fitting model...")
    mdl.fit(X, y, **kwargs)

    print("    --> Scores calculation...")
    X_test, y_test = np.array(data_test.drop([0], axis = 1), dtype = 'float'), np.array(data_test[0])
    if model_name == "KERNEL" : #Faut transformer X
        X_test = apply_kernels(X_test, kernels)

    y_pred = mdl.predict(X_test)

    if model_name == "NN" :
        y_pred = np.array([unique_labels[np.argmax(y)] for y in y_pred], dtype = 'int')

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
    #====>>> /!\/!\/!\ WORKS ONLY FOR BINARY CLASSFICATION !/!\/!\/!\ <<<====
    unique_labels = np.sort( data[0].unique() )
    count_label = np.array( [ len(data[data[0] == label].index) for label in unique_labels ] )
    data_per_class = [data[data[0] == label] for label in unique_labels]

    indice_max = np.argmax(count_label)
    max_label_count, label_max = np.max(count_label), unique_labels[indice_max]
    
    for i in tqdm( range(nb_iteration) ) :
        print_title(" TRAINING of {} (iteration {}/{}) ".format(model_name, i+1, nb_iteration))

        model, kwargs = get_model(model_name = model_name, data = data)


        print("--> Default")
        scores = train(model, data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Default"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Detault_{}".format(i+1)] = scores

    
        print("--> ROS")
        new_data = data_per_class[indice_max].copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "ROS", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count))
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "ROS"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["ROS_{}".format(i+1)] = scores


        print("--> Jittering")
        new_data = data_per_class[indice_max].copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "Jit", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count))
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Jit"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Jit_{}".format(i+1)] = scores
        #Plot TNSE
        if i == 0 :
            plot_tsne(new_data, method = "Jittering", dataset_name = dataset_name ,count_label = count_label, unique_labels = unique_labels)


        print("--> TimeWarping")
        new_data = data_per_class[indice_max].copy()
        for j in range(len(unique_labels)) :
            if unique_labels[j] != label_max :
                data_minor = timeseries_trans( pd.concat( [data_per_class[indice_max] , data_per_class[j] ], axis=0), name_trans = "TW", minor_class = (unique_labels[j] ,count_label[j]), major_class = (label_max, max_label_count))
                new_data = pd.concat([new_data,data_minor], axis=0)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "TW"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["TW_{}".format(i+1)] = scores
        #Plot TNSE
        if i == 0 :
            plot_tsne(new_data, method = "TimeWarping", dataset_name = dataset_name ,count_label = count_label, unique_labels = unique_labels)



        sampling_strategy = {unique_labels[i] : np.max(count_label) for i in range(len(unique_labels))}

        print("--> Basic Smote")
        new_data = timeseries_smote(data , name_trans = "Basic", sampling_strategy = sampling_strategy)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Basic"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Basic_{}".format(i+1)] = scores
        #Plot TNSE
        if i == 0 :
            plot_tsne(new_data, method = "Basic", dataset_name = dataset_name ,count_label = count_label, unique_labels = unique_labels)


        print("--> Basic Adasyn")
        try :
            new_data = timeseries_smote(data , name_trans = "Ada", sampling_strategy = sampling_strategy)
            scores = train(model, new_data, data_test, **kwargs) 
            scores["Model"] = model_name
            scores["Transformation"] = "Ada"
            scores["Dataset"] = dataset_name
            scores_matrix.loc["Ada_{}".format(i+1)] = scores
            #Plot TNSE
            if i == 0 :
                plot_tsne(new_data, method = "Ada", dataset_name = dataset_name ,count_label = count_label, unique_labels = unique_labels)

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
        """



    #pd.set_option('display.max_rows', None)

    return scores_matrix




def make_final_tab(scores_matrix, save_plot_path = "results"):

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

    dataset_folder = "./datasets"
    dataset = "Earthquakes"

    data = pd.read_csv(dataset_folder + "/{}/{}_TRAIN.tsv".format(dataset, dataset) ,sep='\t', header =None)
    data_test = pd.read_csv(dataset_folder + "/{}/{}_TEST.tsv".format(dataset, dataset) ,sep='\t', header =None)


    


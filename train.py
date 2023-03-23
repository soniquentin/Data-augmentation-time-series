from tools import *
from get_info import print_title
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from imblearn.metrics import geometric_mean_score
import seaborn as sns
from models import get_model
from scipy.stats import ttest_ind

#Train and calculate the score
def train(model, new_data, data_test, **kwargs) :

    mdl, model_name = model

    X, y = np.array(new_data.drop([0], axis = 1), dtype = 'float'), np.array(new_data[0])

    print("    --> Fitting model...")
    mdl.fit(X, y, **kwargs)

    print("    --> Scores calculation...")
    X_test, y_test = np.array(data_test.drop([0], axis = 1), dtype = 'float'), np.array(data_test[0])
    y_pred = mdl.predict(X_test)

    if model_name == "NN" :
        y_pred = np.array([round(y[0]) for y in y_pred], dtype = 'int')

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

    label1, label2 = unique_labels[0],unique_labels[1]
    count1, count2 = len( data[data[0] == label1].index), len( data[data[0] == label2].index)
    
    if count1 > count2 :
        minor_class = (label2, count2)
        major_class = (label1, count1)
    else :
        major_class = (label2, count2)
        minor_class = (label1, count1)

    
    for i in range(nb_iteration) :
        print_title(" TRAINING of {} (iteration {}/{}) ".format(model_name, i+1, nb_iteration))
        model, kwargs = get_model(model_name = model_name, data = data)

        print("--> Default")
        scores = train(model, data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Default"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Detault_{}".format(i+1)] = scores

        print("--> ROS")
        new_data = timeseries_trans(data, name_trans = "ROS", minor_class = minor_class, major_class = major_class)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "ROS"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["ROS_{}".format(i+1)] = scores

        print("--> Jittering")
        new_data = timeseries_trans(data, name_trans = "Jit", minor_class = minor_class, major_class = major_class)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Jit"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Jit_{}".format(i+1)] = scores


        print("--> TimeWarping")
        new_data = timeseries_trans(data, name_trans = "TW", minor_class = (1, 3), major_class = (0 , 5))
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "TW"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["TW_{}".format(i+1)] = scores


        print("--> Basic Smote")
        new_data = timeseries_smote(data, name_trans = "Basic", k_neighbors = 2)
        X, y = np.array(new_data.drop([0], axis = 1)), np.array(new_data[0])
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Basic"
        scores["Dataset"] = dataset_name
        scores_matrix.loc["Basic_{}".format(i+1)] = scores


        print("--> Basic Adasyn")
        try :
            new_data = timeseries_smote(data, name_trans = "Ada", k_neighbors = 2)
            scores = train(model, new_data, data_test, **kwargs) 
            scores["Model"] = model_name
            scores["Transformation"] = "Ada"
            scores["Dataset"] = dataset_name
            scores_matrix.loc["Ada_{}".format(i+1)] = scores
        except Exception as e :
            print("    /!\/!\/!\ Asadyn failed /!\/!\/!\ :")
            print("    " + str(e))


    return scores_matrix


def make_final_tab(scores_matrix):

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
        plt.savefig("results/{}_{}_{}.png".format(model_name, metric, dataset_name))
    
    return final_tab_mean, final_tab_p_value

    


if __name__ == "__main__" :

    dataset_folder = "./datasets"
    dataset = "Earthquakes"

    data = pd.read_csv(dataset_folder + "/{}/{}_TRAIN.tsv".format(dataset, dataset) ,sep='\t', header =None)
    data_test = pd.read_csv(dataset_folder + "/{}/{}_TEST.tsv".format(dataset, dataset) ,sep='\t', header =None)
    score_matrix = make_score_test(data, data_test, model_name = "RF", nb_iteration = 20)

    final_tab_mean, final_tab_p_value = make_final_tab(score_matrix)
    


from tools import *
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from get_info import print_title
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from imblearn.metrics import geometric_mean_score
import pickle
import random as rd
import seaborn as sns


def get_model(model_name, data):
    """
        model_name : "RF" (Random Forest), "NN" (Simple fully connected layer)
    """

    if model_name == "RF" :
        random_state = rd.randint(1,100)
        return  RandomForestClassifier(n_estimators = 130,
                                       max_depth = 50, #Set to 50 instead of None to prevent from overfitting
                                       random_state = random_state) , {}
    elif model_name == "NN" :
        nb_timestamp = len(data.columns) - 1

        model = Sequential()
        
        model.add(Dense(64,  input_dim=nb_timestamp))
        model.add(Activation("relu"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(1))
        model.add(Activation("softmax"))

        model.compile(loss='mean_absolute_error', optimizer= Adam(learning_rate = 0.001), metrics=['mean_absolute_error'])

        kwargs = {"epochs" : 100, "batch_size" : 32, "verbose" : 0}

        return model, kwargs


#Train and calculate the score
def train(model, new_data, data_test, **kwargs) :

    X, y = np.array(new_data.drop([0], axis = 1)), np.array(new_data[0])

    print("    --> Fitting model...")
    model.fit(X,y, **kwargs)

    print("    --> Scores calculation...")
    X_test, y_test = np.array(data_test.drop([0], axis = 1)), np.array(data_test[0])
    y_pred = model.predict(X_test)
    y_pred = np.array([round(y) for y in y_pred], dtype = 'int')

    return { "MCC" : matthews_corrcoef(y_test, y_pred), 
            "F1" : f1_score(y_test, y_pred, average = "weighted"), 
            "G-mean" : geometric_mean_score(y_test, y_pred, average = "weighted"), 
            "Acc" :  accuracy_score(y_test, y_pred)}



def make_score_test(data, data_test, model_name = "RF", nb_iteration = 5):
    """
        model_name : "RF" (Random Forest), "NN" (Simple fully connected layer)
        nb_iteration : Nombre d'entrainements par transformation (une moyenne, c'est quand même plus fiable)
    """

    scores_matrix = pd.DataFrame(columns=['MCC','F1','G-mean','Acc', 'Model', "Transformation"])


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
        scores_matrix.loc["Detault_{}".format(i+1)] = scores

        print("--> ROS")
        new_data = timeseries_trans(data, name_trans = "ROS", minor_class = minor_class, major_class = major_class)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "ROS"
        scores_matrix.loc["ROS_{}".format(i+1)] = scores

        print("--> Jittering")
        new_data = timeseries_trans(data, name_trans = "Jit", minor_class = minor_class, major_class = major_class)
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Jit"
        scores_matrix.loc["Jit_{}".format(i+1)] = scores


        print("--> TimeWarping")
        new_data = timeseries_trans(data, name_trans = "TW", minor_class = (1, 3), major_class = (0 , 5))
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "TW"
        scores_matrix.loc["TW_{}".format(i+1)] = scores


        print("--> Basic Smote")
        new_data = timeseries_smote(data, name_trans = "Basic", k_neighbors = 2)
        X, y = np.array(new_data.drop([0], axis = 1)), np.array(new_data[0])
        scores = train(model, new_data, data_test, **kwargs)
        scores["Model"] = model_name
        scores["Transformation"] = "Basic"
        scores_matrix.loc["Basic_{}".format(i+1)] = scores


        print("--> Basic Adasyn")
        try :
            new_data = timeseries_smote(data, name_trans = "Ada", k_neighbors = 2)
            scores = train(model, new_data, data_test, **kwargs) 
            scores["Model"] = model_name
            scores["Transformation"] = "Ada"
            scores_matrix.loc["Ada_{}".format(i+1)] = scores
        except Exception as e :
            print("    /!\/!\/!\ Asadyn failed /!\/!\/!\ :")
            print("    " + str(e))


    with open('scores.pickle', 'wb') as handle:
        pickle.dump(scores_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return scores_matrix


def make_final_tab(scores_matrix):

    all_trans = scores_matrix["Transformation"].unique()
    all_metrics = ["MCC", "F1", "G-mean", "Acc"]

    final_tab = pd.DataFrame(columns= all_metrics)

    for trans in all_trans :
        sub_score_trans = scores_matrix[scores_matrix["Transformation"] == trans].drop(["Model", "Transformation"], axis = 1)
        final_tab.loc[trans] = sub_score_trans.mean(axis = 0)


    for metric in all_metrics :
        plt.figure(figsize=(10, 10), dpi=80) 
        sns.violinplot(data=scores_matrix, x=metric, y="Transformation").set(title=metric)
    
    plt.show()
    
    return final_tab

    


if __name__ == "__main__" :

    dataset_folder = "./datasets"
    dataset = "Earthquakes"

    data = pd.read_csv(dataset_folder + "/{}/{}_TRAIN.tsv".format(dataset, dataset) ,sep='\t', header =None)
    data_test = pd.read_csv(dataset_folder + "/{}/{}_TEST.tsv".format(dataset, dataset) ,sep='\t', header =None)
    score_matrix = make_score_test(data, data_test, model_name = "RF", nb_iteration = 20)

    final_tab = make_final_tab(score_matrix)

    print(final_tab)


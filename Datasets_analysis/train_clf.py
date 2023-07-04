"""
Entraine un classifier pour trouver des corrélations entre datasets et Delta_acc

More infos :
https://www.quentinlao.com/projects/data_augmentation/predictive_model.html
"""
from utils import get_charac_and_metric
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import argparse
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error
import sys, os
from sklearn.model_selection import train_test_split
import random as rd
from scipy.stats import ttest_ind



"""
Classes :
    - Multi : entraine un régresseur pour chaque couple classifier x method DA
    - Single : entraine un régresseur pour tous les couples classifier x method DA
    - Conditionned_Single : entraine un régresseur pour tous les couples classifier x method DA, mais conditionne les données avec le couple en cours

Fonctions :
    - convert_into_data : convertit les données en un format utilisable par les réseaux de neurones
    - normalize_X_y : normalise les données
    - renormalize_y : renormalise les données
    - calc_metric : calcule la métrique INVERSION et MAE
"""


###################
#### MULTI ########
###################

class Multi() :

    def __init__(self) :
        self.trained_models = {}
        self.model_x_DA = []
        

    def train(self, X, y_all, **kwargs) :
        """
            Entraine un régresseur pour chaque couple modèle x méthode

            X et y doivent être pandas DataFrame !
        """

        X = X.to_numpy()
        self.model_x_DA = list(y_all)
        final_loss = 0

        for model_DA in tqdm(self.model_x_DA, desc = "Training model ...") :

            y = y_all[model_DA]
            y = y.to_numpy()

            model = self.get_new_model(X,y)
            final_loss += model.fit(X, y, **kwargs).history["loss"][-1]

            self.trained_models[model_DA] = model

        print(f"Average finale loss : {final_loss/len(self.model_x_DA)}")


    def predict(self, X) :
        """
            X doit être un pandas DataFrame !

            Renvoie un panda dataframe !
        """
        X = X.to_numpy()
        y_pred = pd.DataFrame()


        for model_DA in self.model_x_DA : 
            y_pred[model_DA] = self.trained_models[model_DA].predict(X, verbose = 0).flatten()

        
        return y_pred


    def get_new_model(self, X, y) :
        

        ## MLP
        model = Sequential()

        model.add(Dense(128,  input_dim=X.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(64, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("sigmoid"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        model.add(Activation("relu")) 

        model.compile(loss='mean_absolute_error', optimizer= Adam(learning_rate = 0.001), metrics=['mean_absolute_error'])


        """
        ## Random Forest
        model = RandomForestRegressor(n_estimators = 200, max_depth = 20, random_state = 0)
        """


        return model



###################
#### SINGLE #######
###################


class Single() :

    def __init__(self) :
        pass
        

    def train(self, X, y_all, **kwargs) :
        """
            Entraine un régresseur pour chaque couple modèle x méthode

            X et y doivent être pandas DataFrame !
        """
        self.model_x_DA = list(y_all)

        X = X.to_numpy()
        y_all = y_all.to_numpy()

        self.trained_model = self.get_new_model(X, y_all)

        try : 
            self.trained_model.fit(X, y_all, **kwargs)
        except Exception : 
            self.trained_model.fit(X, y_all)


    def predict(self, X) :
        """
            X doit être un pandas DataFrame !

            Renvoie un panda dataframe !
        """
        X = X.to_numpy()
        y_pred = pd.DataFrame( columns = self.model_x_DA , data = self.trained_model.predict(X, verbose = 0) )

        return y_pred


        

    def get_new_model(self, X, y) :
        

        ## MLP

        model = Sequential()




        model.add(Dense(128,  input_dim=X.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(64, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("sigmoid"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(y.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        model.add(Activation("relu")) 

        model.compile(loss='mean_absolute_error', optimizer= Adam(learning_rate = 0.001), metrics=['mean_absolute_error'])


        ## Random Forest
        """
        model = RandomForestRegressor(n_estimators = 200, max_depth = 20, random_state = 0)
        """


        return model




###################
## CONDITIONNED ###
###################


class Contionned_Single() :
    
    def __init__(self) :
        self.model_x_DA = []
        self.caracs = []


    def condition(self, X,y = None): 
        """
        Transforme X,y (deux panda DataFrame) en de nouveaux X_new, y_new :
        X_new : pandas DataFrame avec les caractéristiques de X et et une one-hot encoding du couple (classifier, method DA) utilisé
        y_new : Numpy array avec le score du couple (classifier, method DA) utilisé
        """
        
        X_new = []
        y_new = []

        if y is None :
            current_y = X
        else :
            current_y = y

        #Pendant l'entrainement, y n'est pas None
        for (index1, row1), (index2, row2) in zip( X.iterrows() , current_y.iterrows() ) : 
            for i,model_x_DA in enumerate(self.model_x_DA) :
                try :
                    sample = [row1[carac] for carac in self.caracs]
                except Exception :
                    raise Exception(f"Ce dataset n'a pas les mêmes caractéristiques que ceux du moodèle")
                sample += [0]*len(self.model_x_DA)
                sample[len(self.caracs) + i] = 1
                X_new.append(sample)
                if y is not None :
                    y_new.append(row2[model_x_DA])
        
        X_new = pd.DataFrame(columns = self.caracs + self.model_x_DA, data = X_new)
        y_new = np.array(y_new)

        """
        X_new["F1 Target"] = y_new
        #Save X in csv
        X_new.to_csv("infos_RAONI.csv")
        exit()
        """


        return X_new, y_new

    def uncondition(self, X_conditionned, y_conditionned_array) :

        """
        Contraire de la fonction condition
        Part d'un X conditionné et y conditionné (même forme que la sortie de la fonction condition)
        et renvoie un X et y non conditionné (même forme que l'entrée de la fonction condition)
        """

        #Concatène les deux dataframe
        X_conditionned["y"] = y_conditionned_array[:,0]

        assert len(X_conditionned)%len(self.model_x_DA) == 0, "Le nombre de lignes de X doit être un multiple du nombre de modèles"


        #On garde que les caractéristiques pour X
        X = []
        y = []
        count_line = 0
        for index, row in X_conditionned.iterrows() :

            #Trouve le model_x_DA correspondant avec le one-hot encoding
            for i in range(len(self.model_x_DA)) :
                if row[self.model_x_DA[i]] == 1 :
                    if count_line%len(self.model_x_DA) == 0 :
                        new_matrix_y = {self.model_x_DA[i] : row["y"]}
                    else :
                        new_matrix_y[self.model_x_DA[i]] = row["y"]
                    break

            if count_line%len(self.model_x_DA) == len(self.model_x_DA) - 1 :
                assert len(new_matrix_y) == len(self.model_x_DA), "Il manque des model_x_DA dans le one-hot encoding"
                y.append(new_matrix_y)
                X.append([row[carac] for carac in self.caracs])

            count_line += 1

        X = pd.DataFrame(columns = self.caracs, data = X)
        y = pd.DataFrame(columns = self.model_x_DA, data = y)

        return X,y
            

    
    def train(self, X, y_all, **kwargs) :

        self.model_x_DA = list(y_all)
        self.caracs = list(X.columns)
        
        X_new, y_new = self.condition(X,y_all)

        self.trained_model = self.get_new_model(X_new, y_new)
        

        #Train et plot la loss
        history = self.trained_model.fit(X_new, y_new, **kwargs)
        """
        plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        """



    def predict(self, X) :

        X_new, _ = self.condition(X)
        y_pred = self.trained_model.predict(X_new, verbose = 0)

        _, y_pred  = self.uncondition(X_new, y_pred)

        return y_pred


    def get_new_model(self, X, y) :


        ## MLP

        model = Sequential()
        
        model.add(Dense(128,  input_dim=X.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("sigmoid"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(64, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        #model.add(BatchNormalization())
        model.add(Activation("sigmoid"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        model.add(Activation("relu")) 

        model.compile(loss='mean_squared_error', optimizer= Adam(learning_rate = 0.001), metrics=['mean_absolute_error'])

        return model



def convert_into_data(global_delta_metric, charac_lists, all_models, all_transfo, target_metric) :
    """
    Convertit les données en un panda dataframe de la forme dont les colonnes sont : 
    X : dataset, carac1, carac2, ..., caracN
    y : model1_method1, model1_method2, ..., model1_methodN, model2_method1, ..., modelN_methodN
    """
    #Récupère la liste des datasets
    datasets = list(global_delta_metric[target_metric].keys())

    #Récupère la liste des caractéristiques
    caracs = list(charac_lists.keys())

    #Crée le panda dataframe
    X = pd.DataFrame(columns = caracs)
    y = pd.DataFrame(columns = [f"{model}_{method}" for model in all_models for method in all_transfo])

    #Remplit le panda dataframe
    index_cursor = 0
    for dataset in datasets :
        charac_dataset = np.array( [charac_lists[carac][dataset] for carac in caracs] )
        X.loc[index_cursor] = [charac_lists[carac][dataset] for carac in caracs]
        y.loc[index_cursor] = [global_delta_metric[target_metric][dataset][model][method] for model in all_models for method in all_transfo]
        index_cursor += 1

    #Remplace Nan par la moyenne de chaque colonne
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    #Check if still Nan
    assert not(X.isnull().values.any()), f"X possède encore des Nan values : {X.isnull().sum().sum()}"
    assert not(y.isnull().values.any()), f"y possède encore des Nan values : {y.isnull().sum().sum()}"

    #Rajoute la colonne Dataset
    X["Dataset"] = datasets

    return X,y


def normalize_X_y(X,y) :
    """
    Normalise X entre -1 et 1 et y entre 0 et 1
    """

    X_max = X.max()
    X_min = X.min()
    X = (X - X_min) / (X_max - X_min)
    X = X * 2 - 1

    y_min = y.min()
    y_max = y.max()
    y = (y - y_min) / (y_max - y_min)

    return X,y, {"X_max" : X_max, "X_min" : X_min, "y_max" : y_max, "y_min" : y_min}


def renormalize_y(y, normalization_dict) :
    """
    Renormalize y
    """
    y_max = normalization_dict["y_max"]
    y_min = normalization_dict["y_min"]

    y = y * (y_max - y_min) + y_min

    return y


def calc_metric(y, y_pred) :
    """
    Calcule la métrique
    """



    #Check que y et y_pred ont la même taille
    assert y.shape == y_pred.shape, f"y et y_pred n'ont pas la même taille : {y.shape} != {y_pred.shape}"




    ### ======== NUMBER OF INVERSION ======== ###
    def mean_inversion(y, y_pred) :
        """
        Calcule le nombre moyen d'inversion
        """

        inversion_list = []

        #Iter sur les lignes
        for (index, row), (index_pred, row_pred) in zip( y.iterrows() , y_pred.iterrows() ) :
            inversion = 0
            weight_sum = 0

            row = row.sort_values() #trie row
            columns = list(row.index)
            max_row = row[columns[-1]]
            min_row = row[columns[0]]

            row_pred = row_pred.sort_values() #trie row_pred
            rank_in_pred = { col : i for i, col in enumerate(row_pred.index) }

            for i in range(len(columns)) :
                for j in range(i+1, len(columns)) :
                    #weight = ( np.sqrt( row[columns[i]]*row[columns[j]])/max_row  )**5
                    weight = (1 - (1 - row[columns[i]]/max_row)*(1 - row[columns[j]]/max_row)  )**20

                    if rank_in_pred[columns[i]] > rank_in_pred[columns[j]] :
                        inversion += weight*1
                        #print(f"Weight : {weight} // row[columns[i]] : {row[columns[i]]} // row[columns[j]] : {row[columns[j]]} // max_row : {max_row}")
                        
                    weight_sum += weight
            
            inversion_list.append(inversion / weight_sum)
        return np.mean(inversion_list)
    mi = mean_inversion(y, y_pred)


    ### ======== MAE ======== ###
    mae = mean_absolute_error(y, y_pred)


    return mae,mi


def main(args) :
    ### ======= Récupère les caractéristiques et les delta_metric ======= ###
    #charac_lists de la forme : {carac1 : {dataset1 : valeur, dataset2 : valeur, ...}, carac2 : {dataset1 : valeur, dataset2 : valeur, ...}, ...}
    #global_delta_metric de la forme : { Metric1 : {dataset1 : {model1 : {method_DA1 : Delta_Acc, method_DA2 : Delta_Acc, ...}, ...} ...} ...}
    global_delta_metric, charac_lists, _, all_models, all_transfo = get_charac_and_metric()
    X, y_brut = convert_into_data(global_delta_metric, charac_lists, all_models, all_transfo, args.target_metric)

    #print(y_brut.head(5))
    X.drop("Dataset", axis=1, inplace=True)

    ### ======= Normalise les données ======= ###
    X, y, normalization_dict = normalize_X_y(X,y_brut)
    nb_row = X.shape[0]


    ### ======= Entraîne le modèle ======= ###
    if args.mode == "multi" :
        model = Multi()

        #Definit le scheduler dans le callback
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 1000 else lr * 0.99)
        
        kwargs = {"epochs" : 700, "batch_size" : 16, "verbose" : 0, "callbacks" : [scheduler_callback]}
    elif args.mode == "single" :
        model = Single()

        #Definit le scheduler dans le callback
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 1000 else lr * 0.99)

        kwargs = {"epochs" : 2000, "batch_size" : 16,  "callbacks" : [scheduler_callback]}
    elif args.mode == "conditionned_single" :
        model = Contionned_Single()
        
        #Definit le scheduler dans le callback
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 400 else lr * 0.94)

        kwargs = {"epochs" : 700, "batch_size" : 16, "callbacks" : [scheduler_callback]}
    
    if args.train == "On" :
        model.train(X.head(nb_row - 5), y.head(nb_row - 5), **kwargs)
        #Sauvegarde model
        with open(f"models/{args.mode}_{args.target_metric}.h5", "wb") as f :
            pickle.dump(model, f)
    else :
        #Charge model
        with open(f"models/{args.mode}_{args.target_metric}.h5", "rb") as f :
            model = pickle.load(f)
    

    ### ======= Prédit les valeurs pour un dataset ======= ###
    y_pred = model.predict(X.tail(5))
    y_pred = renormalize_y(y_pred, normalization_dict) #Renormalise les données
    y_brut = renormalize_y(y.tail(5), normalization_dict) #Renormalise les données
    #print("\n"*3, y_pred)
    mae, mi = calc_metric(y_brut.tail(5), y_pred)

    print(f"MIR : {mi}")
    print(f"MAE : {mae}")


def make_test(args):
    nb_interation = int(args.iteration)
    possible_mode = ["multi", "single", "conditionned_single"]

    mir_list = {mode : [] for mode in possible_mode}
    mae_list = {mode : [] for mode in possible_mode}

    for i in range(nb_interation) :

        print(f"\n\n\n==================== ITERATION {i+1} ====================")

        for mode in possible_mode :

            print(f"\nMode : {mode}")

            ### ======= Récupère les caractéristiques et les delta_metric ======= ###
            #charac_lists de la forme : {carac1 : {dataset1 : valeur, dataset2 : valeur, ...}, carac2 : {dataset1 : valeur, dataset2 : valeur, ...}, ...}
            #global_delta_metric de la forme : { Metric1 : {dataset1 : {model1 : {method_DA1 : Delta_Acc, method_DA2 : Delta_Acc, ...}, ...} ...} ...}
            global_delta_metric, charac_lists, _, all_models, all_transfo = get_charac_and_metric()
            X, y_brut = convert_into_data(global_delta_metric, charac_lists, all_models, all_transfo, args.target_metric)

            #print(y_brut.head(5))
            X.drop("Dataset", axis=1, inplace=True)

            ### ======= Normalise les données ======= ###
            X, y, normalization_dict = normalize_X_y(X,y_brut)

            ### ======= Slit les données en train et test ======= ###
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rd.randint(0,1000))


            ### ======= Entraîne le modèle ======= ###
            if mode == "multi" :
                model = Multi()

                #Definit le scheduler dans le callback
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 1000 else lr * 0.99)
                
                kwargs = {"epochs" : 700, "batch_size" : 16, "verbose" : 0, "callbacks" : [scheduler_callback]}
            elif mode == "single" :
                model = Single()

                #Definit le scheduler dans le callback
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 1000 else lr * 0.99)

                kwargs = {"epochs" : 2000, "batch_size" : 16,  "callbacks" : [scheduler_callback], "verbose" : 0}
            elif mode == "conditionned_single" :
                model = Contionned_Single()
                
                #Definit le scheduler dans le callback
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch < 400 else lr * 0.94)

                kwargs = {"epochs" : 700, "batch_size" : 16, "callbacks" : [scheduler_callback], "verbose" : 0}
            

            model.train(X_train, y_train, **kwargs)


            ### ======= Prédit les valeurs pour un dataset ======= ###
            y_pred = model.predict(X_test)
            y_pred = renormalize_y(y_pred, normalization_dict) #Renormalise les données
            y_test = renormalize_y(y_test, normalization_dict) #Renormalise les données
            #print("\n"*3, y_pred)
            mae, mi = calc_metric(y_test, y_pred)

            mir_list.get(mode).append(mi)
            mae_list.get(mode).append(mae)
        
            print(f"Mir : {mir_list.get(mode)[-1]}")
            print(f"Mae : {mae_list.get(mode)[-1]}")
    

    ## save pandas
    dfs_mode = {}
    for mode in possible_mode :
        print(f"\n\n\n==================== {mode} RESULT ====================")
        df = pd.DataFrame({"MIR" : mir_list.get(mode), "MAE" : mae_list.get(mode)})
        df.to_csv(f"{mode}_{args.target_metric}.csv", index=False)
        print(df.describe(include='all'))
        dfs_mode[mode] = df
    
    ##Make a ttest between all couples of mode
    for mode1 in possible_mode :
        for mode2 in possible_mode :
            if mode1 != mode2 :
                print(f"\n\n\n==================== {mode1} VS {mode2} ====================")
                print(ttest_ind(dfs_mode.get(mode1)["MIR"], dfs_mode.get(mode2)["MIR"]))
                print(ttest_ind(dfs_mode.get(mode1)["MAE"], dfs_mode.get(mode2)["MAE"]))


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="multi, single, conditionned_single, make_test", default="multi")
    parser.add_argument("--train", help="On, Off", default="On")
    parser.add_argument("--target_metric", help="F1, MCC, Acc, G-Mean", default="F1")
    parser.add_argument("--iteration", help="int, useful only for make_test mode", default="20")
    args = parser.parse_args()

    if args.mode == "make_test" :
        make_test(args)
    else :
        main(args)





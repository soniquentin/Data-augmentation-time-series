import pandas as pd
import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns


def import_info() :
    return pd.read_csv("infos.csv" , sep=',')


def m_section(dataset_name):
    #Importe les données
    info = import_info()

    #Cherche le chemin du dataset dans le csv info
    file_path = None
    for index, row in info.iterrows():
        if dataset_name in row["Name"] :
            file_path = row["Path"]
            break
    if file_path == None :
        raise Exception(f"Dataset '{dataset_name}' not found")


    data = pd.read_csv(file_path, sep='\t', header =None)

    unique_labels = data[0].unique()
    
    #Créer un dictionnaire { label : (count, color) } trié par ordre croissant de label
    count_dict_label = { label : len(data[data[0] == label].index) for label in unique_labels}
    count_dict_label = {k: v for k, v in sorted(count_dict_label.items(), key=lambda item: item[1])}
    i = 0
    for k,v in count_dict_label.items():
        count_dict_label[k] = (v, (i/( len(count_dict_label) - 1) , 0,i/( len(count_dict_label) - 1)))
        i+= 1

    
    #On trace un example de chaque label
    fig, axs = plt.subplots(len(count_dict_label))
    j = 0
    for label,v in count_dict_label.items():
        count, color = v
        label_samples = np.array( data[data[0] == label] )[:,1:]
        random_index = rd.randint(0,len(label_samples) - 1)

        axs[j].plot(label_samples[random_index], color = color)
        axs[j].set_title(str(label),loc='right')
        j += 1

    
    plt.show()



def d_section(dataset_name):
    #Importe les données
    info = import_info()

    #Cherche le chemin du dataset dans le csv info
    file_path = None
    for index, row in info.iterrows():
        if dataset_name in row["Name"] :
            file_path = row["Path"]
            break
    if file_path == None :
        raise Exception(f"Dataset '{dataset_name}' not found")


    data = np.array( pd.read_csv(file_path, sep='\t', header =None) )
    length = data.shape[-1] - 1
    df = pd.DataFrame()
    for i in range(1, length + 1) :
        step_df = pd.DataFrame()
        step_df["label"] = data[:,0]
        step_df["step"] = i
        step_df["value"] = data[:,i]

        df = pd.concat( [df, step_df], axis = 0)
    
    sns.lineplot(data = df, x="step", y="value", hue = "label")
    plt.show()




    


def raise_usage():
    usage = """
Run command : python draw.py <DATASET_NAME> <OPTION=--m>

<OPTION> :
    --m : dessine seulement une courbe au hasard par label
    --d : trace la d
            """
    print(usage)
    exit()




if __name__ == "__main__" :

    try :
        dataset_name = sys.argv[1]
    except Exception as e:
        raise_usage()

    try :
        option = sys.argv[2]
    except Exception as e:
        option = "--m"
    

    if "--m" in option : 
        m_section(dataset_name)
    elif "--d" in option :
         d_section(dataset_name)
    else :
        raise_usage()
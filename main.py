import sys
from get_info import get_info
from tools import *
from train import *

def raise_usage():
    usage = """
Run command : python main.py <OPTION>

<OPTION> :
    --get_info_dataset : affiche infos des fichiers d'entrainement (nb de samples, compte pour chaque label, imbalence degree).
    --make_tab <DATASET_NAME> <MODEL>("RF","NN") <NB_ITERATION>
            """
    print(usage)
    exit()


if __name__ == "__main__" :

    try :
        option = sys.argv[1]
    except Exception as e:
        raise_usage()

    if "--get_info_dataset" in option :
        get_info()
    elif "--make_tab" in option :
        try :
            dataset_name = sys.argv[2]
            model_name = sys.argv[3]
            nb_iteration = sys.argv[4]

        except Exception as e:
            raise_usage()  
        
        dataset_folder = "./datasets"

        data = pd.read_csv(dataset_folder + "/{}/{}_TRAIN.tsv".format(dataset_name, dataset_name) ,sep='\t', header =None)
        data_test = pd.read_csv(dataset_folder + "/{}/{}_TEST.tsv".format(dataset_name, dataset_name) ,sep='\t', header =None)
        score_matrix = make_score_test(data, data_test, model_name = model_name, nb_iteration = nb_iteration)

        final_tab = make_final_tab(score_matrix)

        print(final_tab)
        
"""
DATASETS_TO_TEST = [
"ECGFiveDays",
"SonyAIBORobotSurface1",
"Wafer",
"Earthquakes",
"ProximalPhalanxOutlineCorrect",
"ECG200",
"Lightning2",
"PhalangesOutlinesCorrect",
"Strawberry",
"MiddlePhalanxOutlineCorrect",
"HandOutlines",
"DistalPhalanxOutlineCorrect",
"Herring",
"Wine",
"WormsTwoClass",
"Yoga",
"GunPointOldVersusYoung",
"GunPointMaleVersusFemale",
"FordA",
"FordB",
"Computers",
"HouseTwenty",
"TwoLeadECG",
"BeetleFly",
"BirdChicken",
"GunPointAgeSpan",
"ToeSegmentation1",
"GunPoint",
"ToeSegmentation2",
"PowerCons",
"ItalyPowerDemand",
"DodgerLoopWeekend",
"DodgerLoopGame",
"MoteStrain",
"FreezerSmallTrain",
"DodgerLoopWeekend",
"DodgerLoopGame",
"SonyAIBORobotSurface2",
"FreezerRegularTrain",
"ShapeletSim",
"Ham",
"Coffee",
"SemgHandGenderCh2",
"Chinatown"
                    ]
"""

DATASETS_TO_TEST = ["Homemade1"]

#Taille des groupes pour l'entrainement
group_size = 3

#(Model, nb_iterations)
MODELS_TO_TEST = [("NN",20),
                  ("RF",20),
                  ("TS-RF",20),
                  #("DTW_NEIGBOURS",3),
                  ("KERNEL",20),
                  ("SHAPELET",20)
                  ]

summary_metric = ["F1", "G-mean", "Acc", "MCC"]

#Les caractéristiques de dataset dont il faut analyser l'influence (correspond au nom des colonnes dans info.csv)
DATASET_CHARACTERISTICS = [
                           "Length",
                           "Dataset size",
                           "Avg label size",
                           "Dataset variance",
                           "Intra-class variance",
                          ]

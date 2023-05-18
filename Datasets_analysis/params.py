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

#DATASETS_TO_TEST = ["Homemade3"]


DATASETS_TO_TEST = ['Haptics', 'SyntheticControl', 'Worms', 'Computers', 'HouseTwenty',  'Chinatown', 'UWaveGestureLibraryAll', 'Strawberry', 'Car', 'GunPointAgeSpan',  'BeetleFly', 'Wafer', 'CBF', 'Adiac', 'ItalyPowerDemand',  'Trace', 'PigAirwayPressure', 'ShapesAll', 'Beef',  'Mallat', 'GunPointOldVersusYoung', 'MiddlePhalanxTW',  'Meat', 'Herring', 'MiddlePhalanxOutlineCorrect', 'InsectEPGRegularTrain', 'FordA', 'SwedishLeaf', 'InlineSkate',  'UMD', 'CricketY',                'SmoothSubspace', 'OSULeaf', 'Ham', 'CricketX', 'SonyAIBORobotSurface1', 'ToeSegmentation1', 'ScreenType', 'PigArtPressure', 'SmallKitchenAppliances', 'Crop', 'MoteStrain',  'ECGFiveDays', 'Wine', 'SemgHandMovementCh2', 'FreezerSmallTrain', 'UWaveGestureLibraryZ', 'NonInvasiveFetalECGThorax1', 'TwoLeadECG', 'Lightning7', 'Phoneme', 'SemgHandSubjectCh2',  'MiddlePhalanxOutlineAgeGroup',  'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'FacesUCR', 'ECG5000',   'HandOutlines', 'GunPointMaleVersusFemale', 'Coffee', 'Rock', 'MixedShapesSmallTrain',  'FordB', 'FiftyWords', 'InsectWingbeatSound', 'MedicalImages', 'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'ChlorineConcentration', 'Plane', 'ACSF1', 'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGSmallTrain',  'EOGVerticalSignal', 'CricketZ', 'FaceFour', 'RefrigerationDevices',  'MixedShapesRegularTrain', 'GunPoint',  'ECG200', 'ToeSegmentation2', 'WordSynonyms', 'Fungi', 'BirdChicken', 'SemgHandGenderCh2', 'OliveOil', 'BME', 'LargeKitchenAppliances', 'SonyAIBORobotSurface2', 'Lightning2', 'EthanolLevel', 'UWaveGestureLibraryX', 'FreezerRegularTrain', 'Fish', 'ProximalPhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'UWaveGestureLibraryY', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes', 'PowerCons', 'DiatomSizeReduction', 'CinCECGTorso', 'PigCVP', 'ProximalPhalanxTW']



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
                           "Dispersion smoothness",
                           "Bhattacharyya",
                           "Mean smoothness"
                          ]

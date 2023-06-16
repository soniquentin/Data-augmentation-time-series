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
"Chinatown"]
"""

DATASETS_TO_TEST = ["StockMarket"]


#DATASETS_TO_TEST = ['Haptics', 'SyntheticControl', 'Worms', 'Computers', 'HouseTwenty',  'Chinatown', 'UWaveGestureLibraryAll', 'Strawberry', 'Car', 'GunPointAgeSpan',  'BeetleFly', 'Wafer', 'CBF', 'Adiac', 'ItalyPowerDemand',  'Trace', 'PigAirwayPressure', 'ShapesAll', 'Beef',  'Mallat', 'GunPointOldVersusYoung', 'MiddlePhalanxTW',  'Meat', 'Herring', 'MiddlePhalanxOutlineCorrect', 'InsectEPGRegularTrain', 'FordA', 'SwedishLeaf', 'InlineSkate',  'UMD', 'CricketY',                'SmoothSubspace', 'OSULeaf', 'Ham', 'CricketX', 'SonyAIBORobotSurface1', 'ToeSegmentation1', 'ScreenType', 'PigArtPressure', 'SmallKitchenAppliances', 'Crop', 'MoteStrain',  'ECGFiveDays', 'Wine', 'SemgHandMovementCh2', 'FreezerSmallTrain', 'UWaveGestureLibraryZ', 'NonInvasiveFetalECGThorax1', 'TwoLeadECG', 'Lightning7', 'Phoneme', 'SemgHandSubjectCh2',  'MiddlePhalanxOutlineAgeGroup',  'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'FacesUCR', 'ECG5000',   'HandOutlines', 'GunPointMaleVersusFemale', 'Coffee', 'Rock', 'MixedShapesSmallTrain',  'FordB', 'FiftyWords', 'InsectWingbeatSound', 'MedicalImages', 'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'ChlorineConcentration', 'Plane', 'ACSF1', 'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGSmallTrain',  'EOGVerticalSignal', 'CricketZ', 'FaceFour', 'RefrigerationDevices',  'MixedShapesRegularTrain', 'GunPoint',  'ECG200', 'ToeSegmentation2', 'WordSynonyms', 'Fungi', 'BirdChicken', 'SemgHandGenderCh2', 'OliveOil', 'BME', 'LargeKitchenAppliances', 'SonyAIBORobotSurface2', 'Lightning2', 'EthanolLevel', 'UWaveGestureLibraryX', 'FreezerRegularTrain', 'Fish', 'ProximalPhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'UWaveGestureLibraryY', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes', 'PowerCons', 'DiatomSizeReduction', 'CinCECGTorso', 'PigCVP', 'ProximalPhalanxTW']



#Taille des groupes pour l'entrainement
group_size = 3



#(Model, nb_iterations)

MODELS_TO_TEST = [ { "Name" : "NN", "Iterations" : 2, "Make Test" : True },
                   { "Name" : "LSTM", "Iterations" : 2, "Make Test" : False },
                   { "Name" : "RF", "Iterations" : 2, "Make Test" : True },
                   { "Name" : "TS-RF", "Iterations" : 2, "Make Test" : True },
                   { "Name" : "DTW_NEIGBOURS", "Iterations" : 3, "Make Test" : False },
                   { "Name" : "KERNEL", "Iterations" : 2, "Make Test" : True },
                   { "Name" : "SHAPELET", "Iterations" : 2, "Make Test" : True }
                  ]



ALL_TRANSFO = ["ROS", "Jit", "TW", "Basic", "Ada", "GAN", "DTW-SMOTE"] #N'a aucune influence sur les tests, c'est juste à titre indicatif

#Les métriques à utiliser pour l'analyse de la performance
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
                           "Mean smoothness",
                           'DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac', 'CO_HistogramAMI_even_2_5', 'CO_trev_1_num', 'MD_hrv_classic_pnn40', 'SB_BinaryStats_mean_longstretch1', 'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff', 'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat', 'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd', 'SP_Summaries_welch_rect_area_5_1', 'SB_BinaryStats_diff_longstretch0', 'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid', 'FC_LocalSimple_mean3_stderr',
                           "Number of periods",
                           "ID"
                          ]

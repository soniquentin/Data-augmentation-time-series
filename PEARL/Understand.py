import argparse
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow_probability import layers
import seaborn as sns
import time
from utils import get_charac_and_metric
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import ReLU, Dropout
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import pickle

# Batch size
BATCH_SIZE = 16

# Number of training epochs for neural network
EPOCHS = 300

# Learning rate for neural network
L_RATE = 1e-2


def build_model():
    # Multilayer dense neural network
    D = df_encoded_normalized.shape[1]
    model = Sequential([
        Dense(512, use_bias=False, input_shape=(D,)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(256, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(128, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(32, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(1)
    ])
    return model


def train_neural_network(X_train, y_train, X_test, y_test):
    model = build_model()
    model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=L_RATE),
                  loss='mean_absolute_error')
    model.summary()

    # Fit the model
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                    patience=20, min_lr=1e-6)
    ES = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25)
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr, ES],
                        verbose=1)

    return model


def train_xgboost(X_train, y_train, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Neural Network or XGBoost model for prediction.')
    parser.add_argument('--NN', action='store_true', help='Use Neural Network model.')
    parser.add_argument('--XGB', action='store_true', help='Use XGBoost model.')
    parser.add_argument('--RF', action='store_true', help='Use Random Forest model.')
    args = parser.parse_args()

    X = pd.read_csv("data_condi.csv", sep=',')
    y = pd.read_csv("target_condi.csv", sep=',')

    # Normalize the data per feature except the dummy variables
    df_encoded = X.drop(columns=['Dataset'])

    dummy_columns = ['model', 'method']

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Get the subset of the dataframe with only the non-dummy columns
    df_non_dummy = df_encoded.drop(dummy_columns, axis=1)

    # Apply normalization per feature (column) to the non-dummy columns
    df_normalized = pd.DataFrame(scaler.fit_transform(df_non_dummy), columns=df_non_dummy.columns)

    # Combine the normalized non-dummy columns with the original dummy columns
    df_encoded_normalized = pd.concat([df_normalized, df_encoded[dummy_columns]], axis=1)
    df_encoded_normalized = pd.get_dummies(df_encoded_normalized, columns=['model', 'method'])

    y_one = y["F1"]
    X_train, X_test, y_train, y_test = train_test_split(df_encoded_normalized, y_one, test_size=0.2, random_state=42)

    if args.NN:
        model = train_neural_network(X_train, y_train, X_test, y_test)
        model.save("models/model_NN.h5")
        print("Saved model to disk")

    elif args.XGB:
        # Define the hyperparameter grid to search
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.1, 0.2]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree')

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)

        # Perform grid search on the data
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and best model
        params = grid_search.best_params_
        model = grid_search.best_estimator_

        print("Best Hyperparameters:")
        print(params)
        with open("models/model_XGB.pkl", "wb") as f:
                pickle.dump(model, f)
        print("Saved XGBoost model to disk")
    
    
    elif args.RF:
        # Define the hyperparameter grid to search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_model = RandomForestRegressor()

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)

        # Perform grid search on the data
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and best model
        params = grid_search.best_params_
        model = grid_search.best_estimator_
        print("Best Hyperparameters for Random Forest:")
        print(params)
        with open("models/model_RF.pkl", "wb") as f:
                pickle.dump(model, f)
        print("Saved Random Forest model to disk")
            
    else:
        raise ValueError("You must specify either --NN or --XGB.")

  

    y_pred = model.predict(X_test)

    if args.NN:
        y_pred = y_pred.reshape(y_pred.shape[0])

    y_test = y_test.to_numpy()

    # Compute the mean absolute error of the prediction
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"MAE = {mae:.3f}")

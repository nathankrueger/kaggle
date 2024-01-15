import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

base_dir = Path(os.path.dirname(__file__))

TEST_CSV = base_dir / 'test.csv'
TRAIN_CSV = base_dir / 'train.csv'

def get_dataset(csv_path):
    ds = pd.read_csv(csv_path)

    # handle 'na' categorical features
    ds['MiscFeature'].fillna('None', inplace=True)
    ds['PoolQC'].fillna('No Pool', inplace=True)
    ds['Alley'].fillna('No Alley access', inplace=True)
    ds['Fence'].fillna('No Fence', inplace=True)
    ds['MasVnrType'].fillna('None', inplace=True)
    ds['MasVnrType'].fillna('None', inplace=True)
    ds['FireplaceQu'].fillna('No Fireplace', inplace=True)
    ds['GarageType'].fillna('No Garage', inplace=True)
    ds['GarageFinish'].fillna('No Garage', inplace=True)
    ds['GarageQual'].fillna('No Garage', inplace=True)
    ds['GarageCond'].fillna('No Garage', inplace=True)
    ds['BsmtExposure'].fillna('No Basement', inplace=True)
    ds['BsmtFinType1'].fillna('No Basement', inplace=True)
    ds['BsmtFinType2'].fillna('No Basement', inplace=True)
    ds['BsmtCond'].fillna('No Basement', inplace=True)
    ds['BsmtQual'].fillna('No Basement', inplace=True)

    # handle 'na' numeric values
    ds['LotFrontage'].fillna(ds['LotFrontage'].mean(), inplace=True)
    ds['GarageYrBlt'].fillna(0, inplace=True)
    ds['MasVnrArea'].fillna(ds['MasVnrArea'].mean(), inplace=True)

    # drop all rows with missing values (in any column) -- remove rows with missing 'Electrical' column
    ds = ds.dropna()

    # convert string objects to integer category labels
    object_cols = ds.columns.to_series().groupby(ds.dtypes).groups[np.dtype('O')]
    for col in object_cols:
        labeler = preprocessing.LabelEncoder()
        ds[col] = labeler.fit_transform(ds[col])

    ds = ds.astype('float32')

    print(ds.info())
    return ds

def tensorflow_solution():
    train_ds = get_dataset(TRAIN_CSV)
    train_ds_noid = train_ds.drop('Id', axis=1)
    train_ds_noid = train_ds_noid.drop('SalePrice', axis=1)
    train_sales_prices = train_ds.get('SalePrice').to_numpy()
    train_data = train_ds_noid.to_numpy()
    print(train_data.shape)

    # normalize the data
    train_data_mean = train_data.mean(axis=0)
    train_data_std = train_data.std(axis=0)
    train_data -= train_data_mean
    train_data /= train_data_std

    activation = 'tanh'
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(train_data.shape[1],), dtype='float32'),
            keras.layers.Dense(1024, activation=activation),
            keras.layers.Dense(512, activation=activation),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(1, activation=None)
        ]
    )
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss='mse',
        metrics=['mae']
    )

    print(model.summary())

    try:
        model.fit(
            x=train_data,
            y=train_sales_prices,
            batch_size=64,
            validation_split=0.3,
            epochs=300,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ReduceLROnPlateau(
                    patience=6,
                    factor=0.75
                ),
                keras.callbacks.EarlyStopping(
                    patience=30,
                    restore_best_weights=True
                )
            ]
        )
    except KeyboardInterrupt:
        pass

    test_ds = get_dataset(TEST_CSV)
    test_ds_noid = test_ds.drop('Id', axis=1)
    test_data = test_ds_noid.to_numpy()
    test_data -= train_data_mean
    test_data /= train_data_std

    predicitons = model.predict(test_data)
    print(predicitons)

if __name__ == '__main__':
    ds = pd.read_csv(TRAIN_CSV)

    # Provides statistical metrics like mean, stddev, min, max, percentiles, etc
    print("Describing the data:")
    print(ds.describe())

    # Provides information about colums and their types -- f
    print("Dataset Info:")
    print(ds.info())

    # Report on the number of rows with missing data for each column, filtering those that 
    # do not have any rows with missing data.
    missing_data = ds.isnull().sum()
    missing_data = missing_data[missing_data>0]
    print(missing_data.sort_values(ascending=False))

    #print(ds.Alley)
    print(type(ds.get('Alley')[1453]))
    #feature = ds.get('OpenPorchSF')
    #print(feature)

    tensorflow_solution()
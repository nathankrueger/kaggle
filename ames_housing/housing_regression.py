import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os

# prevent ellipsis when describing the data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

base_dir = Path(os.path.dirname(__file__))

TEST_CSV = base_dir / 'test.csv'
TRAIN_CSV = base_dir / 'train.csv'

def get_dataset(csv_path) -> pd.DataFrame:
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
    ds['GarageYrBlt'].fillna(ds['YearBuilt'], inplace=True)
    ds['MasVnrArea'].fillna(ds['MasVnrArea'].mean(), inplace=True)
    ds['BsmtFinSF1'].fillna(0.0, inplace=True)
    ds['BsmtFinSF2'].fillna(0.0, inplace=True)
    ds['BsmtUnfSF'].fillna(0.0, inplace=True)
    ds['TotalBsmtSF'].fillna(0.0, inplace=True)
    ds['BsmtFullBath'].fillna(0.0, inplace=True)
    ds['BsmtHalfBath'].fillna(0.0, inplace=True)
    ds['GarageCars'].fillna(0.0, inplace=True)
    ds['GarageArea'].fillna(0.0, inplace=True)
    
    # create new features
    ds['YearsSinceRemodel'] = ds['YearRemodAdd'] - ds['YearBuilt']

    # convert string objects to integer category labels
    object_cols = ds.columns.to_series().groupby(ds.dtypes).groups[np.dtype('O')]
    for col in object_cols:
        labeler = preprocessing.LabelEncoder()
        ds[col] = labeler.fit_transform(ds[col])

    # fill missing Electrical value for row(s) with missing value
    ds['Electrical'].fillna(ds['Electrical'].mode(), inplace=True)

    ds = ds.astype('float32')
    return ds

def explore_dataset(ds: pd.DataFrame):
    # provides statistical metrics like mean, stddev, min, max, percentiles, etc
    print("Describing the data:")
    print(ds.describe())

    # provides information about colums and their types -- f
    print("Dataset Info:")
    print(ds.info())

    # report on the number of rows with missing data for each column, filtering those that 
    # do not have any rows with missing data.
    missing_data = ds.isnull().sum()
    missing_data = missing_data[missing_data>0]
    print(missing_data.sort_values(ascending=False))

    # access & print some specifics
    print(ds.Alley)
    print(type(ds.get('Alley')[1453]))
    feature = ds.get('OpenPorchSF')
    print(feature)

    # show a heatmap to visualize feature correlations
    plt.figure(figsize=(18, 14))
    heatmap = sns.heatmap(ds.corr(), vmin=-1, vmax=1, xticklabels=True, yticklabels=True, cmap='PiYG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12)
    plt.show()

def keep_only(ds: pd.DataFrame, keep_columns: list) -> pd.DataFrame:
    all_cols = ds.columns.to_list()
    result = ds
    for col in all_cols:
        if not col in keep_columns:
            result = result.drop(col, axis=1)
    
    return result

def tensorflow_solution():
    features_to_model = [
        'Neighborhood',

        # Sale details
        'SaleType',
        'MSZoning',

        # Quality
        'KitchenQual',
        'HeatingQC',
        'GarageQual',
        'CentralAir',
        'OverallQual',
        'OverallCond',
        'ExterQual',
        'ExterCond',
        'PoolQC',
        'FireplaceQu',

        # Basic features of house
        'BldgType',
        'LotConfig',
        'HouseStyle',
        'Bedroom',
        'YearBuilt',
        'YearsSinceRemodel',
        'LotArea',
        'BedroomAbvGr',
        'Fireplaces',
        'GrLivArea',
        'RoofStyle',
        'Condition1',
        'Condition2',
        'LotConfig',
        'LotShape',
        'LandContour',
        'Street',
        'LotFrontage'
    ]

    # SalePrice is the target / expected output, we need it in every case.
    features_to_model += ['SalePrice']

    train_ds = get_dataset(TRAIN_CSV)
    #train_ds = keep_only(train_ds, features_to_model)
    print('Train dataset info: ', end='')
    train_ds.info()

    train_sales_prices = train_ds.get('SalePrice').to_numpy()
    train_ds_no_targets = train_ds.drop('SalePrice', axis=1)

    # create the numpy array for the test data, being careful to omit the targets
    train_data = train_ds_no_targets.to_numpy()
    print('Training data numpy shape: ' + str(train_data.shape))

    # normalize the data
    train_data_mean = train_data.mean(axis=0)
    train_data_std = train_data.std(axis=0)
    train_data -= train_data_mean
    train_data /= train_data_std

    # build a simple feed-forward net
    activation = 'tanh'
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(train_data.shape[1],), dtype='float32'),
            keras.layers.Dense(512, activation=activation),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(256, activation=activation),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(1, activation='linear')
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=100),
        loss='mse',
        metrics=['mae']
    )

    print(model.summary())

    # train the model
    try:
        model.fit(
            x=train_data,
            y=train_sales_prices,
            batch_size=64,
            validation_split=0.25,
            epochs=2000,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ReduceLROnPlateau(
                    patience=8,
                    factor=0.75
                ),
                keras.callbacks.EarlyStopping(
                    patience=200,
                    restore_best_weights=True
                )
            ]
        )
    except KeyboardInterrupt:
        pass

    # prepare the test data
    test_ds = get_dataset(TEST_CSV)
    #test_ds = keep_only(test_ds, features_to_model)
    test_data = test_ds.to_numpy()
    print('Test data numpy shape: ' + str(test_data.shape))

    # normalize the test data, using the factor as the training data
    test_data -= train_data_mean
    test_data /= train_data_std

    #nan_rows = test_ds.isna().any(axis=0)
    #print(nan_rows)

    # inference
    predicitons = model.predict(test_data)

    # generate the prediction submission
    output_offset = 1461
    with open(base_dir / 'housing_submission.csv', 'w') as output_file:
        # write the expected header
        output_file.write(f'Id,SalePrice{os.linesep}')

        # write the predicitons for each house
        for predidx in range(predicitons.shape[0]):
            output_file.write(f'{predidx + output_offset},{predicitons[predidx][0]}{os.linesep}')

if __name__ == '__main__':
    #ds = get_dataset(TRAIN_CSV)
    #explore_dataset(ds)
    tensorflow_solution()
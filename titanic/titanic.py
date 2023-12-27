import keras
import keras_tuner as kt
import tensorflow_decision_forests as tfdf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBRFClassifier
import pandas as pd
import os
import random
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
train_csv = base_dir / 'train.csv'
test_csv = base_dir / 'test.csv'
model_path = str(base_dir / 'titanic_model')
xgboost_device = 'cpu'

CABIN_LETTER_DICT = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, None: 8}
NAME_PREFIX_DICT = {'Mrs.': 1, 'Miss': 2, 'Mr.': 3, 'Dr.': 4, None: 5}
EMBARKMENT_DICT = {'C': 1, 'Q': 2, 'S': 3, '': 4}
DEFAULT_AGE = float('nan')
DEFAULT_FARE = float('nan')

# avg age in training data: 29.69, with 177 missing entries
# avg age in test data: 30.27, with 86 missing entries
# 68% of train.csv survivors are female, this is a good common-sense baseline

def parse_dataset_from_csv(csv_path):
    result = []
    with open(csv_path, 'r') as fh:
        lines = fh.readlines()

        # dispose of header
        lines = lines[1:]
        for row in lines:
            row = row.split(',')
            if len(row) == 0:
                continue
            
            test_data = False
            if len(row) == 12:
                test_data = True
            elif len(row) != 13:
                raise Exception('Unexpected data format.')

            # test data does not come supplied with 'Survived' label
            offset = 1 if test_data else 2

            result.append({
                'id': int(row[0]),
                'survived': None if test_data else int(row[1]),
                'pclass': int(row[offset]),
                'name': f'{row[offset + 1]},{row[offset + 2]}',
                'sex': 0 if row[offset + 3] == 'male' else 1,
                'age': float(row[offset + 4]) if len(row[offset + 4]) > 0 else DEFAULT_AGE,
                'sibsp': int(row[offset + 5]),
                'parch': int(row[offset + 6]),
                'ticket': row[offset + 7],
                'fare': float(row[offset + 8]) if len(row[offset + 8]) > 0 else DEFAULT_FARE,
                'cabin': row[offset + 9],
                'embarked': EMBARKMENT_DICT[row[offset + 10].strip()]
            })
            
    return result

def sklearn_random_forest_solution(train_inputs, train_outputs, test_inputs) -> np.ndarray:
    accuracy_results = {}
    num_folds = 4
    models_to_average = 5

    try:
        n_estimator_params = [10,15,20,25,30,35,40,50,100,150,200,250,500,1000,1500,2000,2500,5000]
        random.shuffle(n_estimator_params)

        for max_depth in [4,5,6,7,8,9,10,15,20,25,None]:
            for n_estimators in n_estimator_params:
                fold_results = []
                for train_split, validation_split in KFold(n_splits=num_folds).split(train_inputs, train_outputs):
                    rando_trees = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=133735
                    )
                    rando_trees.fit(train_inputs[train_split], train_outputs[train_split])
                    predictions = rando_trees.predict(train_inputs[validation_split])

                    fold_acc = accuracy_score(predictions, train_outputs[validation_split])
                    fold_results.append(fold_acc)
                
                average_acc = np.mean(fold_results)
                if average_acc in accuracy_results:
                    accuracy_results[average_acc].append((n_estimators, max_depth))
                else:
                    accuracy_results[average_acc] = [(n_estimators, max_depth)]
                
                print(f'Average across {num_folds} k-folds: [n_estimators:{n_estimators}, max_depth:{max_depth}] -- Accuracy: {average_acc * 100.0:.3f}%')
    except KeyboardInterrupt:
        pass

    model_predictions = []
    model_cnt = 0
    for idx, acc_key in enumerate(sorted(accuracy_results, reverse=True)):
        if model_cnt < models_to_average:
            for model_params in accuracy_results[acc_key]:
                if model_cnt < models_to_average:
                    n_estimators, max_depth = model_params
                    print(f'Model {model_cnt}: {acc_key} -- params: n_estimators:{n_estimators}, max_depth:{max_depth}')
                    rando_trees = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=1337
                    )
                    rando_trees.fit(train_inputs, train_outputs)
                    model_predictions.append(rando_trees.predict(test_inputs))

                    model_cnt += 1
                else:
                    break
        else:
            break
    
    model_predictions = np.asarray(model_predictions)
    model_predictions = np.mean(model_predictions, axis=0)
    return model_predictions

def xgboost_random_forest_solution(train_inputs, train_outputs, test_inputs) -> np.ndarray:
    num_folds = 4
    collected_results = {}

    try:
        for subsample in [0.7,0.75,0.8,0.85,0.9]:
            for max_depth in [3,4,5,6,7,8,9]:
                for n_estimators in [1,10,15,20,25,30,40,50,100,150,200,250,500,1000,2000]:
                    model_acc_results = []
                    for train_split, validation_split in KFold(n_splits=num_folds).split(train_inputs, train_outputs):
                        rfc = XGBRFClassifier(
                            n_estimators=n_estimators,
                            subsample=subsample,
                            max_depth=max_depth,
                            device=xgboost_device
                        )
                        rfc.fit(train_inputs[train_split], train_outputs[train_split])
                        predictions = rfc.predict(train_inputs[validation_split])
                        fold_acc = accuracy_score(predictions, train_outputs[validation_split])
                        model_acc_results.append(fold_acc)

                    average_acc = np.mean(model_acc_results)
                    collected_results[average_acc] = n_estimators, max_depth, subsample
                    print(f'n_estimators: {n_estimators}, max_depth:{max_depth}, subsample: {subsample} -- Avg. Accuracy for {num_folds} folds: {average_acc * 100:.4f}%')

        best_acc = sorted(collected_results, reverse=True)[0]
        n_estimators, max_depth, subsample = collected_results[best_acc]
        print(os.linesep + f'Best Accuracy: {best_acc * 100:.4f}% -- n_estimators: {n_estimators}, max_depth:{max_depth}, subsample: {subsample}')
    
    except KeyboardInterrupt:
        pass

    # fit the model with the best parameters on full training data
    rfc = XGBRFClassifier(
        n_estimators=n_estimators,
        subsample=subsample,
        max_depth=max_depth,
        device=xgboost_device
    )
    rfc.fit(train_inputs, train_outputs)
    predictions = rfc.predict(test_inputs)

    return predictions

def tf_gradient_boosted_trees_solution(train_inputs, train_outputs, test_inputs) -> np.ndarray:
    tuner = tfdf.tuner.RandomSearch(num_trials=50)
    tuner.choice('max_depth', [2,3,4,5,6,7,8,9,10,20,30])
    tuner.choice('num_trees', [10,20,30,40,50,100,200,300,500,1000,1500,2000])
    tuner.choice('subsample', [0.7,0.8,0.85,0.9,1.0])
    model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
    model.fit(train_inputs, train_outputs, validation_split=0.3)
    print(model.summary())

    predictions = model.predict(test_inputs)
    return predictions

def parse_dataset_v2(csv_path):
    features = pd.read_csv(csv_path)
    cols = features.columns
    #print('cols: ' + cols)
    diction = features.to_dict(orient='index')
   # print(diction)
    print(features.describe())

    result = []
    for key in diction:
        result.append(diction[key])
    
   # print(result)

def prepare_dataset_for_model(ds, include_age: bool=False, include_fare: bool=False):
    numeric_inputs = []
    numeric_outputs = []

    for passenger in ds:
        # encode name prefix
        name_int = NAME_PREFIX_DICT[None]
        for key in NAME_PREFIX_DICT.keys():
            if key is not None:
                if key in passenger['name']:
                    name_int = NAME_PREFIX_DICT[key]
                    break

        # encode cabin letter
        cabin_int = CABIN_LETTER_DICT[None]
        for key in CABIN_LETTER_DICT.keys():
            if key is not None:
                if key in passenger['cabin']:
                    name_int = CABIN_LETTER_DICT[key]
                    break
        
        # note: ticket is discarded
        passenger_features = [
                passenger['pclass'],
                passenger['sex'],
                passenger['sibsp'],
                passenger['parch'],
                passenger['embarked'],
                name_int,
                cabin_int
            ]

        if include_age:
            passenger_features.append(passenger['age'])

        if include_fare:
            passenger_features.append(passenger['fare'])

        # build per passenger array of passenger features
        numeric_inputs.append(passenger_features)

        # outputs are solely 'survived'
        numeric_outputs.append(passenger['survived'])

    numeric_outputs = np.array(numeric_outputs, dtype='float32')
    numeric_inputs = np.array(numeric_inputs)

    return numeric_inputs, numeric_outputs

def calculate_mean_for_missing_data(ds, key):
    vals = []
    num_missing = 0
    for entry in ds:
        val = entry[key]
        if val is not None:
            vals.append(val)
        else:
            num_missing += 1

    return np.mean(vals), num_missing

def generate_output_csv(csv_filename, predictions: np.ndarray, offset):
    if predictions.ndim > 1:
        predictions = predictions.squeeze()

    # write output csv
    with open(base_dir / csv_filename, 'w') as fh:
        fh.write('PassengerId,Survived\n')
        for idx, pred in enumerate(predictions):
            fh.write(f'{idx + offset},{1 if pred >= 0.5 else 0}\n')

class TitanicHyperModel(kt.HyperModel):
    def __init__(self, num_inputs: int, kfolds: int=1):
        self.num_inputs = num_inputs
        self.kfolds = kfolds

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        # hyperparams to test
        units = hp.Int(name='units', min_value=64, max_value=256, step=16)
        layers = hp.Int(name='layers', min_value=2, max_value=5, step=1)
        activation = hp.Choice(name='activation', values=['tanh', 'relu'])
        layer_size_reduction = hp.Choice('layer_size_reduction_scheme', values=[0.8, 0.5, 0.3])
        layer_reduction_period = hp.Choice('layer_reduction_period', values=[1, 2])
        optimizer = hp.Choice(name='optimizer', values=['adam', 'rmsprop'])
        use_dropout = hp.Boolean(name='use_dropout', default=True)
        l2_regularization_val = hp.Choice(name='l2_regularization', values=[0.0, 0.001, 0.002, 0.004, 0.008])

        if l2_regularization_val > 0.0:
            l2_regularization = keras.regularizers.l2(l2_regularization_val)
        else:
            l2_regularization = None

        # create a sequential model
        model = keras.Sequential([keras.Input(shape=(self.num_inputs,), dtype='float32')])

        # dynamically shape the hidden layers
        curr_layer_size = units
        hidden_layer_idx = 1
        for _ in range(layers):
            model.add(keras.layers.Dense(curr_layer_size, activation=activation, kernel_regularizer=l2_regularization))
            if hidden_layer_idx % layer_reduction_period == 0:
                curr_layer_size = int(curr_layer_size * layer_size_reduction)
        
        if use_dropout:
            model.add(keras.layers.Dropout(0.5))

        # add the output
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=optimizer
        )

        return model
    
    def __get_average_metric_values(self, histories):
        best_results = {}
        total_epochs = 0
        for hist in histories:
            total_epochs += len(hist.epoch)
            for key in hist.history:
                if 'accuracy' in key:
                    if not key in best_results:
                        best_results[key] = []
                    best_results[key].append(np.amax(hist.history[key]))
                elif 'loss' in key:
                    if not key in best_results:
                        best_results[key] = []
                    best_results[key].append(np.amin(hist.history[key]))

        best_results_avg = {}
        for key in best_results:
            best_results_avg[key] = np.mean(best_results[key])
        
        result = histories[0]
        for key in result.history:
            if key in best_results_avg:
                # this is destroying history -- just repeat the averaged value to keep the lists of the correct order,
                # and to direct the hypertuning to use the result of k-fold validation across multiple train / test splits
                result.history[key] = [best_results_avg[key]] * total_epochs
            else:
                # this is destroying history -- just repeat 'something' valid to keep the lists of the correct order
                result.history[key] = [result.history[key][0]] * total_epochs
        
        result.epoch = range(1, total_epochs + 1)
        return result

    def fit(self, hp, model, *args, **kwargs):
        if self.kfolds > 1:
            histories = []
            inputs = kwargs['x']
            outputs = kwargs['y']
            for train, validation in KFold(n_splits=self.kfolds).split(inputs, outputs):
                kwargs['x'] = inputs[train]
                kwargs['y'] = outputs[train]
                kwargs['validation_data'] = (inputs[validation], outputs[validation])
                if 'validation_split' in kwargs:
                    del kwargs['validation_split']
                histories.append(super().fit(hp, model, *args, **kwargs))

            # build a combined history by averaging values from each fold...
            return self.__get_average_metric_values(histories)
        else:
            return super().fit(hp, model, *args, **kwargs)

"""
Train a simple model with fixed architecture
"""
def run_single_model(
    train_inputs,
    train_outputs,
    test_inputs,
    monitor: str='val_accuracy',
    use_regularization: bool=False,
    dropout: float=0.0
) -> np.ndarray:
    # Note: https://www.pinecone.io/learn/batch-layer-normalization/

    # multi-layer perceptron
    x = inputs = keras.Input(shape=(len(train_inputs[0]),), dtype='float32')
    x = keras.layers.Dense(128, activation='tanh', kernel_regularizer=keras.regularizers.l1(0.002) if use_regularization else None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='tanh', kernel_regularizer=keras.regularizers.l1(0.002) if use_regularization else None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.002) if use_regularization else None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001) if use_regularization else None)(x)
    if dropout > 0.0:
         x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # train the model
    try:
        history = model.fit(
            batch_size=64,
            validation_split=0.3,
            x=train_inputs,
            y=train_outputs,
            epochs=400,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.5,
                    monitor=monitor
                ),
                keras.callbacks.EarlyStopping(
                    patience=50,
                    monitor=monitor,
                    restore_best_weights=True
                )
            ]
        )
    except KeyboardInterrupt:
        print(os.linesep)

    # best weights are used to make predictions
    predictions = model.predict(test_inputs)
    return predictions

"""
Use keras-tuner to sweep across hyperparameter space and determine the best
configuration hyperparameters for a performant model.
See: https://keras.io/guides/keras_tuner/getting_started/
"""
def run_keras_tuner_on_hypermodel(
    train_inputs,
    train_outputs,
    test_inputs,
    overridden_hp: kt.HyperParameters=None,
    epochs: int=300,
    kfolds: int=3,
    max_trials: int=400,
    monitor: str='val_accuracy',
    tuner_type: str='random'
) -> np.ndarray:

    hypermodel = TitanicHyperModel(len(train_inputs[0]), kfolds=kfolds)
    execution_per_trial = 1
    val_split = 0.3
    batch_sz = 32
    early_stopping_patience = 20
    reduce_lr_patience = 6
    reduce_lr_factor = 0.6

    if tuner_type == 'bayesian':
        tuner = kt.tuners.BayesianOptimization(
            hypermodel=hypermodel,
            objective=monitor,
            max_trials=max_trials,
            executions_per_trial=execution_per_trial,
            directory=base_dir / 'titanic_hypertuning',
            overwrite=True,
            hyperparameters=overridden_hp
        )
    elif tuner_type == 'random':
        tuner = kt.tuners.RandomSearch(
            hypermodel=hypermodel,
            objective=monitor,
            max_trials=max_trials,
            executions_per_trial=execution_per_trial,
            directory=base_dir / 'titanic_hypertuning',
            overwrite=True,
            hyperparameters=overridden_hp
        )
    elif tuner_type == 'grid':
        tuner = kt.tuners.GridSearch(
            hypermodel=hypermodel,
            objective=monitor,
            max_trials=max_trials,
            executions_per_trial=execution_per_trial,
            directory=base_dir / 'titanic_hypertuning',
            overwrite=True,
            hyperparameters=overridden_hp
        )
    else:
        raise Exception('Unsupported tuner type specified.')

    print(tuner.search_space_summary())

    try:
        tuner.search(
            x=train_inputs,
            y=train_outputs,
            epochs=epochs,
            validation_split=val_split,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    patience=reduce_lr_patience,
                    factor=reduce_lr_factor
                ),
                keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=early_stopping_patience
                )
            ],
            batch_size=batch_sz
        )
    except KeyboardInterrupt:
        print(os.linesep)

    # print out the top 5 hyperparameter sets
    best_hps = tuner.get_best_hyperparameters(5)
    for idx, best_hp in enumerate(best_hps):
        print(f'#{idx+1} Best HyperParameter values: {os.linesep}{best_hp.values}')

    # choose and build the best model
    best_model = hypermodel.build(best_hps[0])

    # train the best model architecture (best hyperparams)
    history = best_model.fit(
        x=train_inputs,
        y=train_outputs,
        validation_split=val_split,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=reduce_lr_patience,
                factor=reduce_lr_factor
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience
            ),
            keras.callbacks.ModelCheckpoint(
                save_best_only=True,
                filepath=model_path,
                monitor=monitor
            )
        ],
        batch_size=batch_sz
    )

    # load in the best performing model weights
    best_model = keras.models.load_model(model_path)

    # evaluate on training data to confirm loading in the 'best' model was effective
    print(os.linesep + 'Evaluating best model...')
    best_model.evaluate(train_inputs, train_outputs)

    predictions = best_model.predict(test_inputs)
    return predictions

def average_predicitons(*args):
    nones_removed = [arg for arg in args if arg is not None]
    squeezed = [arg.squeeze() if arg.ndim > 1 else arg for arg in nones_removed]
    return np.mean(squeezed, axis=0)

if __name__ == '__main__':
    # parse & collect the training data in a usable format
    train_ds = parse_dataset_from_csv(train_csv)
    random.seed(1337)
    random.shuffle(train_ds)
    train_inputs, train_outputs = prepare_dataset_for_model(train_ds)
    train_inputs_with_nan, train_outputs_with_nan = prepare_dataset_for_model(train_ds, include_age=True, include_fare=True)

    # parse & collect the test data
    test_ds = parse_dataset_from_csv(test_csv)
    test_inputs, _ = prepare_dataset_for_model(test_ds)
    test_inputs_with_nan, _ = prepare_dataset_for_model(test_ds, include_age=True, include_fare=True)
    test_id_offset = int(test_ds[0]['id'])

    hyper_model_predictions = None
    single_model_predictions = None
    sklearn_random_forest_predicitons = None
    xgboost_random_forest_predictions = None
    tf_gradient_boosted_trees_predicitons = None

    # hyper-model NN which first chooses the best model architecture,
    # then builds the model, fits it, and generates predicitons
    # overridden_hp = kt.HyperParameters()
    # overridden_hp.Fixed('activation', 'tanh')
    # overridden_hp.Int(name='layers', min_value=3, max_value=5, step=1)
    # hyper_model_predictions = run_keras_tuner_on_hypermodel(
    #     train_inputs,
    #     train_outputs,
    #     test_inputs,
    #     monitor='val_accuracy',
    #     tuner_type='random',
    #     max_trials=4096,
    #     kfolds=3,
    #     epochs=100,
    #     overridden_hp=overridden_hp
    # )

    # single NN model which is fit and generates predicitons
    single_model_predictions = run_single_model(
        train_inputs,
        train_outputs,
        test_inputs,
        monitor='val_accuracy',
        dropout=0.0,
        use_regularization=False
    )

    # random forest example which ensembles (combines) the results of multiple random forests
    sklearn_random_forest_predicitons = sklearn_random_forest_solution(train_inputs, train_outputs, test_inputs)

    # random forest which chooses the best model given hyperparameter tuning
    xgboost_random_forest_predictions = xgboost_random_forest_solution(train_inputs_with_nan, train_outputs_with_nan, test_inputs_with_nan)

    # gradient boosted trees with tensorflow & hyperparameter tuning
    tf_gradient_boosted_trees_predicitons = tf_gradient_boosted_trees_solution(train_inputs_with_nan, train_outputs_with_nan, test_inputs_with_nan)

    # averaged predicitons among different methods
    averaged_predictions = average_predicitons(
                                hyper_model_predictions,
                                single_model_predictions,
                                sklearn_random_forest_predicitons,
                                xgboost_random_forest_predictions,
                                tf_gradient_boosted_trees_predicitons
                            )

    # save the results
    generate_output_csv('test_submission.csv', averaged_predictions, test_id_offset)
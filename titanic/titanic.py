import keras
import kerastuner as kt
import numpy as np
import os
import random
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
train_csv = base_dir / 'train.csv'
test_csv = base_dir / 'test.csv'

CABIN_LETTER_DICT = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, None: 8}
NAME_PREFIX_DICT = {'Mrs.': 1, 'Miss': 2, 'Mr.': 3, 'Dr.': 4, None: 5}
EMBARKMENT_DICT = {'C': 1, 'Q': 2, 'S': 3, '': 4}
DEFAULT_AGE = 0.0

# avg age in training data: 29.69, with 177 missing entries
# avg age in test data: 30.27, with 86 missing entries

def parse_dataset(csv_path):
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
                'fare': float(row[offset + 8]) if len(row[offset + 8]) > 0 else None,
                'cabin': row[offset + 9],
                'embarked': EMBARKMENT_DICT[row[offset + 10].strip()]
            })
            
    return result

def prepare_dataset_for_model(ds):
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

        numeric_inputs.append(
            [
                passenger['pclass'],
                passenger['sex'],
                #passenger['age'],
                passenger['sibsp'],
                passenger['parch'],
                passenger['embarked'],
                name_int,
                cabin_int,
                #passenger['ticket'],
                #passenger['fare']
            ]
        )
        numeric_outputs.append(passenger['survived'])

    numeric_outputs = np.array(numeric_outputs)
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

def generate_output_csv(csv_filename, predictions, offset):
    # write output csv
    with open(base_dir / csv_filename, 'w') as fh:
        fh.write('PassengerId,Survived\n')
        for idx, pred in enumerate(predictions):
            fh.write(f'{idx + offset},{1 if pred[0] >= 0.5 else 0}\n')

class TitanicHyperModel(kt.HyperModel):
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        # hyperparams to test
        units = hp.Int(name='units', min_value=64, max_value=256, step=32, default=192)
        layers = hp.Int(name='layers', min_value=2, max_value=5, step=1, default=4)
        activation = hp.Choice(name='activation', values=['tanh', 'relu'], default='tanh')
        layer_size_reduction = hp.Choice('layer_size_reduction_scheme', values=[0.5, 0.3], default=0.3)
        layer_reduction_period = hp.Choice('layer_reduction_period', values=[1, 2], default=1)
        optimizer = hp.Choice(name='optimizer', values=['adam', 'rmsprop'], default='adam')
        use_dropout = hp.Boolean(name='use_dropout', default=True)
        l2_regularization_val = hp.Choice(name='l2_regularization', values=[0.0, 0.001, 0.002, 0.003], default=0.0)

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

"""
Train a simple model with fixed architecture
"""
def run_single_model(train_inputs, train_outputs, use_regularization=False, use_dropout=False) -> keras.Model:
    # multi-layer perceptron
    x = inputs = keras.Input(shape=(len(train_inputs[0]),), dtype='float32')
    x = keras.layers.Dense(96, activation='tanh', kernel_regularizer=keras.regularizers.l1(0.002) if use_regularization else None)(x)
    x = keras.layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.002) if use_regularization else None)(x)
    #x = keras.layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001) if use_regularization else None)(x)
    #x = keras.layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001 if use_regularization else None))(x)
    if use_dropout:
        x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    model_path = str(base_dir / 'titanic_model')

    # train the model
    try:
         history = model.fit(
            batch_size=48,
            validation_split=0.3,
            x=train_inputs,
            y=train_outputs,
            epochs=400,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ModelCheckpoint(
                    save_best_only=True,
                    filepath=model_path,
                    #monitor='val_accuracy'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=15,
                    factor=0.6
                ),
                keras.callbacks.EarlyStopping(
                    patience=30
                )
            ]
        )
    except KeyboardInterrupt:
        print(os.linesep)

    # load in the best performing model
    best_model = keras.models.load_model(model_path)
    return best_model

"""
Use keras-tuner to sweep across hyperparameter space and determine the best
configuration hyperparameters for a performant model.
See: https://keras.io/guides/keras_tuner/getting_started/
"""
def run_keras_tuner_on_hypermodel(train_inputs, train_outputs, max_trials: int=400, monitor: str='val_accuracy') -> keras.Model:
    hypermodel = TitanicHyperModel(len(train_inputs[0]))
    val_split = 0.3
    batch_sz = 48
    epochs = 300

    tuner = kt.tuners.BayesianOptimization(
        hypermodel=hypermodel,
        objective=monitor,
        max_trials=max_trials,
        executions_per_trial=1,
        directory=base_dir / 'titanic_hypertuning',
        overwrite=True
    )

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
                    patience=8,
                    factor=0.6
                ),
                keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=15
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
                patience=8,
                factor=0.6
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=15
            )
        ],
        batch_size=batch_sz
    )

    return best_model
    
if __name__ == '__main__':
    # parse & collect the training data in a usable format
    train_ds = parse_dataset(train_csv)
    random.seed(1337)
    random.shuffle(train_ds)
    numeric_inputs, numeric_outputs = prepare_dataset_for_model(train_ds)

    # convert training data to float
    train_inputs = np.array(numeric_inputs).astype('float32')
    train_outputs = np.array(numeric_outputs).astype('float32')

    #model = run_keras_tuner_on_hypermodel(train_inputs, train_outputs, monitor='val_loss', max_trials=128)
    model = run_single_model(train_inputs, train_outputs, use_dropout=False, use_regularization=False)
  
    # make the predictions
    test_ds = parse_dataset(test_csv)
    test_id_offset = int(test_ds[0]['id'])
    test_inputs, numeric_outputs = prepare_dataset_for_model(test_ds)
    predictions = model.predict(test_inputs)
    generate_output_csv('test_submission.csv', predictions, test_id_offset)
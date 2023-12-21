import keras
import numpy as np
import os
import random
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
train_csv = base_dir / 'train.csv'
test_csv = base_dir / 'test.csv'

NAME_PREFIX_DICT = {'Mrs.': 1, 'Miss': 2, 'Mr.': 3, 'Dr.': 4, None: 5}
EMBARKMENT_DICT = {'C': 1, 'Q': 2, 'S': 3, '': 4}
DEFAULT_AGE = 0
DEFAULT_FARE = 0

def isint(item):
    try:
        int(item)
    except ValueError:
        return False
    return True

def get_encoded_ticket(ticketstr: str) -> int:
    if isint(ticketstr):
        return int(ticketstr)
    else:
        result = 0
        parts = ticketstr.split()
        for part in parts:
            if isint(part):
                result += int(part)
        return result

def parse_dataset(csv_path):
    result = []
    with open(csv_path, 'r') as fh:
        lines = fh.readlines()
        lines = lines[1:] # dispose of header
        for row in lines:
            row = row.split(',')
            if len(row) == 0:
                continue
            
            test_data = False
            if len(row) == 12:
                test_data = True
            elif len(row) != 13:
                raise Exception('Unexpected data format.')

            offset = 1 if test_data else 2

            result.append({
                'id': int(row[0]),
                'survived': None if test_data else int(row[1]),
                'pclass': int(row[offset]),
                'name': f'{row[offset + 1]},{row[offset + 2]}',
                'sex': 0 if row[offset + 3] == 'male' else 1,
                'age': int(float(row[offset + 4])) if len(row[offset + 4]) > 0 else DEFAULT_AGE,
                'sibsp': int(row[offset + 5]),
                'parch': int(row[offset + 6]),
                'ticket': get_encoded_ticket(row[offset + 7]),
                'fare': int(float(row[offset + 8])) if len(row[offset + 8]) > 0 else DEFAULT_FARE,
                'cabin': row[offset + 9],
                'embarked': EMBARKMENT_DICT[row[offset + 10].strip()]
            })
            
    return result

def prepare_dataset_for_model(ds):
    int_inputs = []
    int_outputs = []
    for passenger in ds:
        # encode name prefix
        name_int = NAME_PREFIX_DICT[None]
        for key in NAME_PREFIX_DICT.keys():
            if key is not None:
                if key in passenger['name']:
                    name_int = NAME_PREFIX_DICT[key]

        int_inputs.append(
            [
                passenger['pclass'],
                passenger['sex'],
                passenger['age'],
                passenger['sibsp'],
                passenger['parch'],
                passenger['embarked'],
                name_int,
               # passenger['ticket'],
               # passenger['fare']
            ]
        )
        int_outputs.append(passenger['survived'])

    int_outputs = np.array(int_outputs)
    int_inputs = np.array(int_inputs)
    return int_inputs, int_outputs

if __name__ == '__main__':
    train_ds = parse_dataset(train_csv)
    random.seed(1337)
    random.shuffle(train_ds)
    int_inputs, int_outputs = prepare_dataset_for_model(train_ds)

    val_percentage = 0.3
    total_inputs = len(int_inputs)
    num_val = int(val_percentage * total_inputs)
    num_train = total_inputs - num_val

    # collect training data
    train_inputs = int_inputs[:num_train]
    train_outputs = int_outputs[:num_train]
    train_inputs = np.array(train_inputs).astype('float32')
    train_outputs = np.array(train_outputs).astype('float32')

    # set aside validation data
    val_inputs = int_inputs[num_train:]
    val_outputs = int_outputs[num_train:]
    val_inputs = np.array(val_inputs).astype('float32')
    val_outputs = np.array(val_outputs).astype('float32')

    # simple multi-layer perceptron
    x = inputs = keras.Input(shape=(len(int_inputs[0]),), dtype='float32')
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='tanh')(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    model_path = str(base_dir / 'titanic_model')

    # train the model
    try:
        history = model.fit(
            batch_size=32,
            x=train_inputs,
            y=train_outputs,
            validation_data=(val_inputs, val_outputs),
            epochs=150,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ModelCheckpoint(
                    save_best_only=True,
                    filepath=model_path,
                    #monitor='val_accuracy'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.5
                )
            ]
        )
    except KeyboardInterrupt:
        print(os.linesep)

    # load in the best performing model
    best_model = keras.models.load_model(model_path)
    
    # make the predictions
    test_ds = parse_dataset(test_csv)
    test_id_offset = int(test_ds[0]['id'])
    test_inputs, int_outputs = prepare_dataset_for_model(test_ds)
    predictions = model.predict(test_inputs)

    # write output csv
    with open(base_dir / 'test_submission.csv', 'w') as fh:
        fh.write('PassengerId,Survived\n')
        for idx, pred in enumerate(predictions):
            fh.write(f'{idx + test_id_offset},{1 if pred[0] >= 0.5 else 0}\n')
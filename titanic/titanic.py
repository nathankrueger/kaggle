import tensorflow as tf
import keras
import numpy as np
import os
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
train_csv = base_dir / 'train.csv'
test_csv = base_dir / 'test.csv'

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

            plcass_offset = 1 if test_data else 2

            id = int(row[0])
            if not test_data:
                survived = int(row[1])
            else:
                survived = None
            
            pclass = int(row[plcass_offset])
            name = f'{row[plcass_offset + 1]},{row[plcass_offset + 2]}'
            sex = 0 if row[plcass_offset + 3] == 'male' else 1
            age = int(float(row[plcass_offset + 4])) if len(row[plcass_offset + 4]) > 0 else DEFAULT_AGE
            sibsp = int(row[plcass_offset + 5])
            parch = int(row[plcass_offset + 6])
            ticket = get_encoded_ticket(row[plcass_offset + 7])
            fare = int(float(row[plcass_offset + 8])) if len(row[plcass_offset + 8]) > 0 else DEFAULT_FARE
            cabin = row[plcass_offset + 9]
            embarked = EMBARKMENT_DICT[row[plcass_offset + 10].strip()]
            
            # id is index + 1
            result.append({
                'id': id,
                'survived': survived,
                'pclass': pclass,
                'name': name,
                'sex': sex,
                'age': age,
                'sibsp': sibsp,
                'parch': parch,
                'ticket': ticket,
                'fare': fare,
                'cabin': cabin,
                'embarked': embarked
            })
            
    return result

def prepare_dataset_for_model(ds):
    int_inputs = []
    int_outputs = []
    for passenger in ds:
        int_inputs.append(
            [
                passenger['pclass'],
                passenger['sex'],
                passenger['age'],
                passenger['sibsp'],
                passenger['parch'],
                passenger['embarked'],
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
    int_inputs, int_outputs = prepare_dataset_for_model(train_ds)

    # set aside validation data
    val_percentage = .3
    total_inputs = len(int_inputs)
    num_val = int(val_percentage * total_inputs)
    num_train = total_inputs - num_val

    train_inputs = int_inputs[:num_train]
    train_outputs = int_outputs[:num_train]
    train_inputs = np.array(train_inputs).astype('float32')
    train_outputs = np.array(train_outputs).astype('float32')

    val_inputs = int_inputs[num_train:]
    val_outputs = int_outputs[num_train:]
    val_inputs = np.array(val_inputs).astype('float32')
    val_outputs = np.array(val_outputs).astype('float32')

    x = inputs = keras.Input(shape=(len(int_inputs[0]),), dtype='float32')
    #x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
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

    try:
        history = model.fit(
            batch_size=32,
            x=train_inputs,
            y=train_outputs,
            validation_data=(val_inputs, val_outputs),
            epochs=100,
            callbacks=[
                keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs')),
                keras.callbacks.ModelCheckpoint(
                    save_best_only=True,
                    filepath=model_path
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=8,
                    factor=0.5
                )
            ]
        )
    except KeyboardInterrupt:
        print(os.linesep)

    best_model = keras.models.load_model(model_path)

    test_id_offset = 892
    test_ds = parse_dataset(test_csv)
    test_inputs, int_outputs = prepare_dataset_for_model(test_ds)
    predictions = model.predict(test_inputs)

    # write output csv
    with open(base_dir / 'test_submission.csv', 'w') as fh:
        fh.write('PassengerId,Survived\n')
        for idx, pred in enumerate(predictions):
            fh.write(f'{idx + test_id_offset},{1 if pred[0] >= 0.5 else 0}\n')
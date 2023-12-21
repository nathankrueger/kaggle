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

def get_dataset(csv_path):
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
            print(id)
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
            fare = int(float(row[plcass_offset + 8])) if len(row[plcass_offset + 4]) > 0 else DEFAULT_FARE
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

if __name__ == '__main__':
    ds = get_dataset(train_csv)

    int_inputs = []
    outputs = []
    for passenger in ds:
        int_inputs.append(
            [
                passenger['pclass'],
                passenger['sex'],
                passenger['age'],
                passenger['sibsp'],
                passenger['parch'],
                passenger['embarked'],
                passenger['ticket'],
                passenger['fare']
            ]
        )
        outputs.append(passenger['survived'])

    outputs = np.array(outputs)

    # set aside validation data
    val_percentage = .3
    total_inputs = len(int_inputs)
    num_val = int(val_percentage * total_inputs)
    num_train = total_inputs - num_val
    train_inputs = int_inputs[:num_train]
    train_inputs = np.array(train_inputs)
    val_inputs = int_inputs[num_train:]
    val_inputs = np.array(val_inputs)

    inputs = keras.Input(shape=(len(int_inputs[0]),), dtype=tf.int32)
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    history = model.fit(
        x=train_inputs,
        y=outputs,
        validation_data=val_inputs,
        epochs=50,
        callbacks=[
            keras.callbacks.TensorBoard(str(base_dir / 'tensorboard_logs'))
        ]
    )
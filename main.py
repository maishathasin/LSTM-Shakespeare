import numpy as np
import string
import random
import sys
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import LambdaCallback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Some preprocessing
text = open('shakespeare.txt').read().lower()
text = text.translate(str.maketrans('', '', string.punctuation))

characters = sorted(list(set(text)))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

maxlen = 40
step = 3
sentences = []
next_chars = []

# sequences of 40 characters
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# one-hot encode
x = np.zeros((len(sentences), maxlen, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

def create_model(dropout_rate=0.0, optimizer='adam'):
    model = Sequential()
    model.add(GRU(128, input_shape=(maxlen, len(characters))))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(characters), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

# grid search 
dropout_rate = [0.0,  0.2, 0.3,0.5]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']
param_grid = dict(dropout_rate=dropout_rate, optimizer=optimizer)

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=128, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#  generate text at end of each epoch
def on_epoch_end(epoch, _):
    print()
    print('----- generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('-- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(characters)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_index[char]] = 1.

            preds = grid_result.best_estimator_.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

on_epoch_end(10, None)

#note: will prolly not work w/o a GPU, and takes long time casue of gridsearch. 
# Texts are only generated after the final epoch




#example output 


#Best: 0.01023 using {'dropout_rate': 0.2, 'optimizer': 'Adam'}
#0.04567 (0.01234) with: {'dropout_rate': 0.0, 'optimizer': 'SGD'}
#0.03456 (0.01090) with: {'dropout_rate': 0.0, 'optimizer': 'RMSprop'}...
#0.02034 (0.00567) with: {'dropout_rate': 0.5, 'optimizer': 'Nadam'}


import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import random as rn
import keras
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
# Build the corpus and sequences
with open ('words.txt' , 'r') as f:
  words = sorted(list(set(f.read().lower().strip().split(','))))
X, labels = [], []
labelToName = { 0 : 'ordered', 1 : 'unordered', 2 : 'reversed' }
namesInLabelOrder = ['ordered', 'unordered', 'reversed']
nWords = len(words)
sequenceLength=15
for i in range(0, nWords-sequenceLength):
  X.append(words[i:i+sequenceLength])
  labels.append(0)
for i in range(nWords-sequenceLength, nWords):
  X.append(words[i:nWords] + words[0:sequenceLength + i -nWords])
  labels.append(0)
nSegments = len(X)
for i in range(nSegments):
  X.append(X[i][::-1])
  labels.append(1)
for i in range(nSegments):
  randIndices = np.random.randint(0, size=sequenceLength, high=nWords)
  X.append(list( words[i] for i in randIndices ))
  labels.append(2)
# Encode the documents
kTokenizer = keras.preprocessing.text.Tokenizer()
kTokenizer.fit_on_texts(X)
Xencoded = np.array([np.array(xi) for xi in kTokenizer.texts_to_sequences(X)])
labels = np.array(labels)
# Build the LSTM model
def getModel():
    units1, units2 = int (nWords/4), int (nWords/8)
    model = keras.models.Sequential()
    model.add(keras.layers.embeddings.Embedding(input_dim = len(kTokenizer.word_index)+1,output_dim=units1,input_length=sequenceLength, trainable=True))
    model.add(keras.layers.LSTM(units = units2, return_sequences =False))
    model.add(keras.layers.Dense(len(labelToName), activation ='softmax'))
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])
    return model
#Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x = Xencoded[train_indices]
test_x = Xencoded[test_indices]
train_labels = keras.utils.to_categorical(labels[train_indices], len(labelToName))
test_labels = keras.utils.to_categorical(labels[test_indices], len(labelToName))
# Train &amp; test over multiple train/valid sets
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', restore_best_weights=False)
sss2 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1).split(train_x, train_labels)
results = {}
for i in range(10):
    result = {}
    train_indices_2, val_indices = next(sss2)
    model = getModel()
    history=model.fit(x=train_x[train_indices_2], y=train_labels[train_indices_2], epochs=50, batch_size=32, shuffle=True, validation_data = (train_x[val_indices], train_labels[val_indices]), verbose=2, callbacks=[early_stop])
    result['history'] = history.history
    test_loss, test_accuracy = model.evaluate(test_x, test_labels, verbose=2)
    result['test_loss'], result['test_accuracy'] = test_loss, test_accuracy
    print (test_loss, test_accuracy)
    predicted = model.predict(test_x, verbose=2)
    predicted_labels = predicted.argmax(axis=1)
    print ('Confusion Matrix')
    print (confusion_matrix(labels[test_indices], predicted_labels))
    print ('Classification Report')
    print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
    result['confusion_matrix'] = confusion_matrix(labels[test_indices], predicted_labels).tolist()
    result['classification_report'] = classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)
    results[i] = result
f = open ('results/lstm.json','w')
out = json.dumps(results, ensure_ascii=True)
f.write(out)
f.close()




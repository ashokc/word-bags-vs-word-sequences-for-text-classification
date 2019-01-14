import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random as rn
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
# Build the corpus and sequences
with open ('words.txt' , 'r') as f:
  words = sorted(list(set(f.read().lower().strip().split(','))))
X, labels = [], []
labelToName = { 0 : 'ordered', 1 : 'reversed', 2 : 'unordered' }
namesInLabelOrder = ['ordered', 'reversed', 'unordered']
nWords = len(words)
sequenceLength= 15
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
# get encoded documents
X=np.array([np.array(xi) for xi in X])
labels = np.array(labels)
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
Xencoded=vectorizer.transform(X)
# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x = Xencoded[train_indices]
test_x = Xencoded[test_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]
# Train & test over multiple train/valid sets
results = {}
model = MultinomialNB()
model.fit(train_x, train_labels)
predicted_labels = model.predict(test_x)
results['confusion_matrix'] = confusion_matrix(labels[test_indices], predicted_labels).tolist()
results['classification_report'] = classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)
print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
f = open ('results/nb.json','w')
out = json.dumps(results, ensure_ascii=True)
f.write(out)
f.close()


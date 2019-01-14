import numpy as np
import glob, os, re, sys, json
import matplotlib.pyplot as plt
from PIL import Image

#get_ipython().magic('matplotlib inline')

def plotF1Scores (result):
    f1_scores = []
    for i in range(10):
        f1_scores.append(result[str(i)]['classification_report']["weighted avg"]["f1-score"])
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    width = 0.25
    xVals = np.linspace(1, 10, num=10, dtype=int)
    tick_labels = np.linspace(1, 10, num=10, dtype=int)
    subplot.bar(xVals,f1_scores,width,tick_label=tick_labels)
#    subplot.bar(xVals,f1_scores,width)
#    subplot.xaxis.set_ticks([])
#    subplot.xaxis.set_ticklabels([])
    subplot.set_ylim(bottom=min(f1_scores)-0.01, top=max(f1_scores)+0.01)
    fig.savefig('results/lstm-f1-scores.png', format='png', dpi=720)


def plotConvergence (results):
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    epochs = list(range(0,len(results['val_acc'])))
    subplot.plot(epochs,results['val_acc'],color='g', label='Validation')
    subplot.plot(epochs,results['val_loss'],color='g')
    subplot.plot(epochs,results['acc'],color='b')
    subplot.plot(epochs,results['loss'],color='b', label='Training')
    subplot.legend(loc='upper right', prop={'size': 10})
    fig.savefig('results/lstm-loss-accuracy.png', format='png', dpi=720)

def main():
    with open ('./results/lstm.json') as fh:
        result = json.loads(fh.read())
    plotConvergence (result['7']['history'])
    plotF1Scores (result)

if __name__ == '__main__':
    main()


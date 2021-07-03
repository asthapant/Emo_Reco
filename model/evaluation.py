import itertools
from sklearn.metrics import confusion_matrix,classification_report

emotion = os.listdir('/content/drive/MyDrive/traindataset')
print(emotion)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(7)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    thresh = cm.max() / 2.
    print(thresh)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print(len(validation_generator.classes))
predictions = model.predict(validation_generator)

y_pred = np.argmax(predictions, axis=1)
print(y_pred)

# confusion matrix
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
np.set_printoptions(precision=2)

# non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=emotion,
                      title='Confusion matrix, without normalization')

# normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=emotion, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

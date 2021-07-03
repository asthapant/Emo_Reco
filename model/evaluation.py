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

for i in range(0,10):
  os.makedirs('/content/drive/My Drive/kfold/'+str(i))
  os.makedirs('/content/drive/My Drive/kfold/'+str(i)+'/train')
  os.makedirs('/content/drive/My Drive/kfold/'+str(i)+'/test')
  for j in os.listdir('/content/drive/My Drive/dataset/CK+48'):
    os.makedirs('/content/drive/My Drive/kfold/'+str(i)+'/train/'+j)
    os.makedirs('/content/drive/My Drive/kfold/'+str(i)+'/test/'+j)
    
for i in range(0,10):
  totalfold='/content/drive/My Drive/kfold/'+str(i)
  testfold = '/content/drive/My Drive/kfold/' + str(i) + '/test'
  trainfold = '/content/drive/My Drive/kfold/' + str(i) + '/train'
  for j in os.listdir(trainfold):
    emotion_train_fold = trainfold + '/' + j
    emotion_test_fold = testfold + '/' + j
    emotion_source = '/content/drive/My Drive/dataset/CK+48/' + j
    length = len(os.listdir(emotion_source))
    initial_size = int(i*length/10)
    final_size = int((i+1)*length/10)
    files = []
    for k in os.listdir(emotion_source):
      path = emotion_source + '/' + k 
      files.append(k)
    testing_set = files[initial_size:final_size]
    training_set = []
    for n in files:
      if n not in testing_set:
        training_set.append(n)
    for filename in training_set:
      src = emotion_source + '/' + filename
      des = emotion_train_fold + '/' + filename
      copyfile(src,des)
    for filename in testing_set:
      src = emotion_source + '/' + filename
      des = emotion_test_fold + '/' + filename
      copyfile(src,des)
tot_accuracy=0
for i in range(0,10):
  fold_path = '/content/drive/My Drive/kfold/' + str(i)
  TRAINING_DIR = train_fold = '/content/drive/My Drive/kfold/' + str(i) + '/train'
  train_datagen = ImageDataGenerator(rescale=1./255,
      horizontal_flip=True,
      rotation_range=2,                          
      preprocessing_function=ppreprocessing
      )
  train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=50,
                                                    class_mode='categorical',
                                                    target_size=(100,100),
                                                    shuffle=True,
                                                    color_mode='grayscale')

  VALIDATION_DIR = '/content/drive/My Drive/kfold/' + str(i) + '/test'
  validation_datagen = ImageDataGenerator(
    rescale=1./255,
      horizontal_flip=True,
      rotation_range=2,
      preprocessing_function=ppreprocessing
      )
  validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=50,
                                                              class_mode='categorical',
                                                              target_size=(100,100),
                                                              shuffle=True,
                                                    color_mode='grayscale')
  validation_generator.shuffle = False
  validation_generator.index_array = None
  model = func1_256_neuron(7)
  model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
  history = model.fit(train_generator,epochs=10,batch_size=50,shuffle=True)
  test_loss,test_acc=model.evaluate(validation_generator)
  tot_accuracy=tot_accuracy+test_acc
avg_acc=tot_accuracy/10
print(avg_acc)

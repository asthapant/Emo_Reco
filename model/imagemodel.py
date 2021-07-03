import random
from shutil import copyfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import stats


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('file is zero length, so ignoring')
    
    n_files = len(all_files)
    training_len = int(n_files * SPLIT_SIZE)
    testing_len = int(len(all_files) - training_len)
    shuffled = random.sample(all_files, n_files)
    
    train_set = shuffled[0:training_len]
    test_set = shuffled[:testing_len]
    
    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)
        
        
 split_size=0.8
anger_train_dir = '/content/drive/MyDrive/traindataset/anger/'
sadness_train_dir = '/content/drive/MyDrive/traindataset/sadness/'
disgust_train_dir = '/content/drive/MyDrive/traindataset/disgust/'
happy_train_dir = '/content/drive/MyDrive/traindataset/happy/'
fear_train_dir = '/content/drive/MyDrive/traindataset/fear/'
contempt_train_dir = '/content/drive/MyDrive/traindataset/contempt/'
surprise_train_dir = '/content/drive/MyDrive/traindataset/surprise/'

anger_test_dir = '/content/drive/MyDrive/testdataset/anger/'
sadness_test_dir = '/content/drive/MyDrive/testdataset/sadness/'
disgust_test_dir = '/content/drive/MyDrive/testdataset/disgust/'
happy_test_dir = '/content/drive/MyDrive/testdataset/happy/'
fear_test_dir = '/content/drive/MyDrive/testdataset/fear/'
contempt_test_dir = '/content/drive/MyDrive/testdataset/contempt/'
surprise_test_dir = '/content/drive/MyDrive/testdataset/surprise/'

anger_source_dir = '/content/drive/My Drive/dataset/CK+48/anger/'
sadness_source_dir = '/content/drive/My Drive/dataset/CK+48/sadness/'
disgust_source_dir = '/content/drive/My Drive/dataset/CK+48/disgust/'
happy_source_dir = '/content/drive/My Drive/dataset/CK+48/happy/'
fear_source_dir = '/content/drive/My Drive/dataset/CK+48/fear/'
contempt_source_dir = '/content/drive/My Drive/dataset/CK+48/contempt/'
surprise_source_dir = '/content/drive/My Drive/dataset/CK+48/surprise/

split_data(anger_source_dir,anger_train_dir,anger_test_dir,split_size)
split_data(sadness_source_dir,sadness_train_dir,sadness_test_dir,split_size)
split_data(disgust_source_dir,disgust_train_dir,disgust_test_dir,split_size)
split_data(happy_source_dir,happy_train_dir,happy_test_dir,split_size)
split_data(fear_source_dir,fear_train_dir,fear_test_dir,split_size)
split_data(contempt_source_dir,contempt_train_dir,contempt_test_dir,split_size)
split_data(surprise_source_dir,surprise_train_dir,surprise_test_dir,split_size)
        

def func1_no_neuron(classes):
   model=tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),
   tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(classes,activation='softmax')
                                   
  ])
   return model

def func1_256_neuron(classes):
   model=tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),
   tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(256, activation = 'relu', name = "full_connected_1"),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(classes,activation='softmax')
                                   
  ])
   return model

def func1_512_neuron(classes):
   model=tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),
   tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(512, activation = 'relu', name = "full_connected_1"),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(classes,activation='softmax')
                                   
  ])
   return model
def func1_1024_neuron(classes):
   model=tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(100,100,1)),
   tf.keras.layers.MaxPooling2D((2,2)),
   tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
   tf.keras.layers.MaxPooling2D((2,2)),

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(1024, activation = 'relu', name = "full_connected_1"),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(classes,activation='softmax')
                                   
  ])
   return model

model= func1_256_neuron(7)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


TRAINING_DIR = "/content/drive/MyDrive/traindataset"
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

VALIDATION_DIR = "/content/drive/MyDrive/testdataset"
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

history = model.fit(train_generator,epochs=10,batch_size=50,shuffle=True,validation_data=validation_generator)

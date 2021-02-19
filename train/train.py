import numpy as np
# !pip install tensorflowjs

#Load data
images = np.load("../input/hand-landmarks/images_notflipped.npy")
labels = np.load("../input/hand-landmarks/labels_notflipped.npy")

unique_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

from sklearn.model_selection import train_test_split
from tensorflow import keras

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, stratify = labels)

# Preprocess data
X_train_flatten = np.reshape(X_train, (len(X_train),-1) ) #reshape: n. of tranig samples with 63 features (21*3)
X_test_flatten = np.reshape(X_test, (len(X_test),-1) )
# One hot encoding
y_train_one_hot = keras.utils.to_categorical(y_train)
y_test_one_hot = keras.utils.to_categorical(y_test)


# Idea from: https://keras.io/examples/keras_recipes/quasi_svm/
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
    
model = keras.Sequential()
#model.add(RBFLayer(63, 0.05))
model.add(keras.Input(shape=(63,))) #number of features
model.add(RandomFourierFeatures(output_dim=4096, scale=10.0, kernel_initializer="laplacian"))
model.add(layers.Dense(units=24)) #number of output classes [, kernel_regularizer=keras.regularizers.l2(l2=1/1000)]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

model.fit(X_train_flatten, y_train_one_hot, epochs=20, batch_size=128, validation_split=0.2)

y_pred = model.predict(X_test_flatten)
result = model.evaluate(X_test_flatten, y_test_one_hot)
print(result)


import tensorflowjs as tfjs
# Try to save the model as Keras model
#tfjs.converters.save_keras_model(model, "./model/")
# Save the model as Graph model
model.save("./temp/", overwrite=True)
tfjs.converters.convert_tf_saved_model("./temp/", "./model/")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import itertools
import numpy as np

#Confusion matrix
#Helper function to plot confusion matrix
def plot_confusion_matrix(y, y_pred, title, normalize=False):
    if y.ndim>1:
        y = np.argmax(y, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)
    cm = confusion_matrix(y, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (24, 20))
    ax = plt.subplot()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
    plt.colorbar()
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.set(title = title)
    limit = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] < 0.01:
            plt.text(j, i, format(0, 'd'), horizontalalignment = "center",color = "black")
        else:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center",color = "white" if cm[i, j] > limit else "black")        
    
    plt.show()
    
        
plot_confusion_matrix(y_test_one_hot, y_pred, title='Confusion matrix For Keras Model', normalize=True)
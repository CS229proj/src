from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation

def run_keras():
   model = Sequential()
   model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
   model.add(Activation("relu"))
   model.add(Dense(output_dim=10, init="glorot_uniform"))
   model.add(Activation("softmax"))
   model.compile(loss='categorical_crossentropy', optimizer='sgd')
   model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

if __name__ == "__main__":
   print('hello')

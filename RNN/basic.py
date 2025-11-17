from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

model = Sequential()

model.add(SimpleRNN(3,input_shape=(4,5)))
model.add(Dense(1,activation='sigmoid'))

# print(model.summary())

print(model.get_weights()[0].shape)
print(model.get_weights()[1].shape)
print(model.get_weights()[3].shape)
print(model.get_weights()[4].shape)
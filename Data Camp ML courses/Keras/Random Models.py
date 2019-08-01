
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
###REGRESSION 
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# Specify the model
def get_new_model ():
    n_cols = x_train.shape[1]
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    


#model = get_new_model()
#print("Loss function: " + model.loss)
#model.fit(x_train, y_train,epochs=1)

#Optimizing the model
from keras.optimizers import SGD
lr_to_test = [.000001, .000005, .000007]
early_stopping_monitor = EarlyStopping(patience=2)
 
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics=['mse'])#For Categories and softmax metrics=['accuracy'])
    model.fit(x_train, y_train,epochs=100, validation_split=0.3,callbacks=[early_stopping_monitor])

###Classification 
#
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.utils import to_categorical
#
#target = to_categorical(df.survived)
#model = Sequential()
#model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
#model.add(Dense(2, activation='softmax'))
#model.compile(optimizer='sgd', 
#              loss='categorical_crossentropy', 
#              metrics=['accuracy'])
#
## Fit the model
#model.fit(predictors, target)
#
#predictions = model.predict(x_test)
#predicted_prob_true = predictions[:,1]
#
## print predicted_prob_true
#print(predicted_prob_true)


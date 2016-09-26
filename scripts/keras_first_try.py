from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



# load
dataset = numpy.loadtxt("../anagrafica_files/list4Keras_100k_train", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:290]
Y = dataset[:,290]


#Importing my data generated for validation

dataset_val = numpy.loadtxt("../anagrafica_files/list4Keras_100k_valid", delimiter=",")
# split into input (X) and output (Y) variables
X_val = dataset_val[:,0:290]
Y_val = dataset_val[:,290]

# create model
model = Sequential()
model.add(Dense(300, input_dim=290, init='uniform', activation='relu'))
model.add(Dense(290, init='uniform', activation='relu'))
model.add(Dense(290, init='uniform', activation='relu'))
#model.add(Dense(290, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, nb_epoch=70, batch_size=10,validation_data=(X_val,Y_val))


# evaluate the model
#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#validation, my test

# calculate predictions

predictions = model.predict(X_val,verbose=1)
#print(predictions)
#print(len(predictions))
# round predictions
rounded = [round(x) for x in predictions]
#print(rounded)
test=[round(x) for y in Y_val]
#print('-------------------------------------')
#print(Y_val)





print("Testing prediction power")
right=0
wrong=0
i=0
while i< len(Y_val):
    if int(rounded[i])!=int(Y_val[i]):
        wrong+=1
    else:
        right+=1
    i+=1
print("right:",right)
print("wrong:",wrong)
print("Perentual:",right/(right+wrong))

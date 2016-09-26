import os
import sys
from getopt import getopt
import logging
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy
from keras.models import model_from_json
import random

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger("START")

##########################################################################
### usage
##########################################################################
def usage(msg):
   if msg=='0':
      print("=========================")
      print("ERRORE: %s" % 'Missing variabile in input')
      print("=========================")
#   print( "Usage: %s -i  input-file\n" % sys.argv[0]
   print("Usage: %s -f file input " % sys.argv[0])
   print("Usage: %s -o file output" % sys.argv[0])
   print("Usage: %s -k number of fold for cross validation" % sys.argv[0])
   print("Usage: %s -e number of epoch" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example :python keras_second_try.py -f ../anagrafica_files/files4keras/list4Keras_10k -o ../anagrafica_files/files4keras/results_logloss -k 10 -e 70  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,numFold,numEpoch):

    # load
    print(inFile)
    dataset = numpy.loadtxt(inFile, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:290]
    Y = dataset[:,290]
    #k-fold cross validation
    print(type(numFold))
    kfold = StratifiedKFold(y=Y, n_folds=numFold, shuffle=True, random_state=seed)
    cvscores = []
    csvpred = []

    for i, (train, test) in enumerate(kfold):
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
        model.fit(X[train], Y[train], nb_epoch=numEpoch, batch_size=10)
        #   model.fit(X, Y, nb_epoch=70, batch_size=10,validation_data=(X_val,Y_val)) in quello sopra NON faccio il conto sul validation ad ogni epoch
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        # calculate predictions
        predictions = model.predict(X[test],verbose=1)
        # round predictions
        rounded = [round(x) for x in predictions]
        print("Testing prediction power for run %i"%i)
        right=0
        wrong=0
        j=0
        while j < len(Y[test]):
            if int(rounded[j])!=int(Y[test][j]):
                wrong+=1
            else:
                right+=1
            j+=1
        perc_right=float(right)/float((right+wrong))
        print("right:",right)
        print("wrong:",wrong)
        print("Percentual:",perc_right)
        csvpred.append(perc_right)

#printing out mean and std
    print('Results for evaluate')
    print("%.2f%% (+/- %.2f%%)")% (numpy.mean(cvscores), numpy.std(cvscores))
    print('Results for predictions')
    print("%.2f%% (+/- %.2f%%)" )% (numpy.mean(csvpred), numpy.std(csvpred))

    with open(outFile, "w") as fo:
        fo.write('Results for evaluate\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(cvscores), numpy.std(cvscores)))
        fo.write('Results for predictions\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(csvpred), numpy.std(csvpred)))


# serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")



if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"f:k:o:e:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
       outFile=str(opts['-o'])
    if '-k' in opts:
        numFold=int(opts['-k'])
    if '-e' in opts:
        numEpoch=int(opts['-e'])
    if '-h' in opts:
       usage('msg')
    if ('-f' not in opts==True and '-o' not in opts==True and '-k' not in opts==True and '-e' not in opts==True and '-h' not in opts==True):
        usage('0')
    main(inFile,outFile,numFold,numEpoch)

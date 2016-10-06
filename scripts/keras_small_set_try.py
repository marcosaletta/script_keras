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
   print("Example :python %s -f ../anagrafica_files/files4keras/list4Keras_10k -o ../anagrafica_files/files4keras/results_logloss -k 10 -e 70  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,numFold,numEpoch):

    # load
    logging.info("START")
    logging.info("LOADING THE DATA SET")
#    print(inFile)
    dataset = numpy.loadtxt(inFile, delimiter=",",skiprows=1)
    # split into input (X) and output (Y) variables
    len=dataset.shape[1]
    X = dataset[:,0:len-1]
    Y = dataset[:,len-1]
    #k-fold cross validation
#    print(type(numFold))
    logging.info("SPLITTING SAMPLE FOR CROSS-VALIDATION WITH %i FOLD"%numFold)
    kfold = StratifiedKFold(y=Y, n_folds=numFold, shuffle=True, random_state=seed)
    cvscores = []
    csvpred = []
    logging.info("STARTING WITH CROSS-VALIDATION")
    epoch_num=0
    for i, (train, test) in enumerate(kfold):
        logging.info("START EPOCH NUM %i"%epoch_num)
        epoch_num+=1
        # create model
        model = Sequential()
        model.add(Dense(len+10, input_dim=len-1, init='uniform', activation='relu'))
        model.add(Dense(len-1, init='uniform', activation='relu'))
        model.add(Dense(len-1, init='uniform', activation='relu'))

        #model.add(Dense(290, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X[train], Y[train], nb_epoch=numEpoch, batch_size=10)
        #   model.fit(X, Y, nb_epoch=70, batch_size=10,validation_data=(X_val,Y_val)) in quello sopra NON faccio il conto sul validation ad ogni epoch
        # evaluate the model
        logging.info("RESULT OF RUN N. %i WITH EVALUATE:"%epoch_num)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        # calculate predictions
        """
        predictions = model.predict(X[test],verbose=1)
        # round predictions
#        rounded = [round(x) for x in predictions]
        rounded = numpy.around(predictions.astype(numpy.double),0)
        logging.info("MY TEST OF PREDICTION POWER FOR RUN N. %i"%epoch_num)
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
        csvpred.append(perc_right)}
        """
#printing out mean and std
    logging.info("TOTAL RESULTS")
    logging.info("MEAN AND STANDARD DEVIATION")
    logging.info('RESULTS FOR EVALUATE')
    print("%.2f%% (+/- %.2f%%)")% (numpy.mean(cvscores), numpy.std(cvscores))
#    logging.info('RESULTS FOR MY TEST')
#    print("%.2f%% (+/- %.2f%%)" )% (numpy.mean(csvpred), numpy.std(csvpred))

    with open(outFile, "w") as fo:
        fo.write('Results for evaluate\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(cvscores), numpy.std(cvscores)))
#        fo.write('Results for predictions\n')
#        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(csvpred), numpy.std(csvpred)))


# serialize model to JSON
    logging.info("SAVING MODEL IN model.json AND model.h5")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    logging.info("Saved model to disk")
    logging.info("END")




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
    if '-f' not in opts and '-o' not in opts and '-k' not in opts and '-e' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,numFold,numEpoch)

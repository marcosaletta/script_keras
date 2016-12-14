import os
import sys
import logging
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy
from keras.models import model_from_json
import random
import argparse
from datetime import datetime



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-i','--infile', type=str, help=('csv with data in input'), required=True)
argparser.add_argument('-o','--outfile', type=str, help=('csv with results in output'), required=True)
argparser.add_argument('-f','--fold', type=int, help=('Number of folds'), required=True)
argparser.add_argument('-e','--epoch', type=int, help=('Number of epoch'), required=True)
argparser.add_argument('-p',"--pred", help="Make evalutaion of the model also using predictions",action="store_true")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger("START")

#####################################################################


def LoadDataset(inFile):
    dataset = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13])
    label = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=[0],dtype=numpy.str)
    #dataset = numpy.genfromtxt(inFile, delimiter="|",skiprows=1,dtype=None)#{'names': ('kwd', 'm1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12', 'evergreen'),'formats': ('S1','f2','f2','f2','f2','f2','f2','f2','f2','f2','f2','f2','f2','i1')})
    # split into input (X) and output (Y) variables
    print(dataset.shape)
    len=dataset.shape[1]
    X = dataset[:,0:len-1]
    Y = dataset[:,len-1]
    print(X.shape)
    return X, Y, len

def KFolder(numFold,Y):
    return StratifiedKFold(y=Y, n_folds=numFold, shuffle=True, random_state=seed)

def CreateModel(cvscores,epoch_num,train,test,len,X,Y,numEpoch,model2save):
#    logging.info("EPOCH NUMBER: %i"%epoch_num)
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
    scores_perc=scores[1]*100
    print("%s: %.2f%%" % (model.metrics_names[1], scores_perc))
    if epoch_num==1:
        model2save=model
    else:
        if scores_perc>max(cvscores):
            model2save=model
    cvscores.append(scores_perc)
    return cvscores, model2save


def ModelPredictions(csvpred):
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
    csvpred.append(perc_right)
    return csvpred

def PrintResults(cvscores,csvpred,MakePred,outFile):
    logging.info("TOTAL RESULTS")
    logging.info("MEAN AND STANDARD DEVIATION")
    logging.info('RESULTS FOR EVALUATE')
    print("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))
    logging.info("WRITING RESULTS TO THE OUTFILE: %s"%outFile)
    with open(outFile, "w") as fo:
        fo.write('Results for evaluate\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(cvscores), numpy.std(cvscores)))
        if MakePred:
            logging.info('RESULTS FOR MY TEST')
            print("%.2f%% (+/- %.2f%%)" )% (numpy.mean(csvpred), numpy.std(csvpred))
            fo.write('Results for predictions\n')
            fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(csvpred), numpy.std(csvpred)))

def SaveBestModel(model2save,outFile):
    logging.info("SAVING MODEL IN %s model.json AND %s"%(outFile+"_model.json",outFile+"_model.h5"))
    model_json = model2save.to_json()
    with open(outFile+"_model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model2save.save_weights(outFile+"_model.h5")
    logging.info("Saved model to disk")


def main(argv):
    start_time = datetime.now()
    logging.info("START")
    args = argparser.parse_args()
    inFile = args.infile
    outFile = args.outfile
    numFold = args.fold
    numEpoch = args.epoch
    MakePred=args.pred
    logging.info("LOADING THE DATA SET")
    X,Y,len = LoadDataset(inFile)
    logging.info("SPLITTING SAMPLE FOR CROSS-VALIDATION WITH %i FOLD"%numFold)
    kfold = KFolder(numFold,Y)
    cvscores = []
    csvpred = []
    model2save=0
    logging.info("STARTING WITH CROSS-VALIDATION")
    epoch_num=0
    for i, (train, test) in enumerate(kfold):
        epoch_num+=1
        logging.info("START EPOCH NUM %i"%epoch_num)
# create model
        cvscores, model2save = CreateModel(cvscores,epoch_num,train,test,len,X,Y,numEpoch,model2save)
        if MakePred:
            csvpred=ModelPredictions(csvpred)
#printing out mean and std
    PrintResults(cvscores,csvpred,MakePred,outFile)
# serialize model to JSON
    SaveBestModel(model2save,outFile)
    logging.info("EXECUTED IN %f SEC"%((datetime.now()-start_time)).total_seconds())
    logging.info("END")




if __name__=="__main__":
    main(sys.argv)

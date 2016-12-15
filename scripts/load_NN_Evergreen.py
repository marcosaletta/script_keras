import os
import sys
import logging
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy
from keras.models import model_from_json
import random
from datetime import datetime
import time
import argparse
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-i','--infile',type=str,help=("CSV with the data to be analized"),required=True)
argparser.add_argument('-o','--outfile',type=str,help=("CSV output"),required=True)
argparser.add_argument('-w','--weights',type=str,help=("h5 file with the weights of the model"),required=True)
argparser.add_argument('-j','--json',type=str,help=("json file of the model"),required=True)
argparser.add_argument('-e',"--eval", help="Make evalutaion of the prediction if the csv in input has the fileds of the real values",action="store_true")



def LoadModel(modelJson,modelWeight):
    logging.info("LOADING MODEL")
    with open(modelJson, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    logging.info("LOADING WEIGHTS IN NEW MODEL")
    loaded_model.load_weights(modelWeight)
    logging.info("LOADED MODEL FROM FILE")
    #evaluating performance on the set
    logging.info("COMPILING MODEL")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info("COMPILED")
    return loaded_model

def LoadDataset(inFile,makeEval):
    label = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=[0],dtype=numpy.str)
    if makeEval:
        dataset = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=range(1,13))
        leng=dataset.shape[1]
        X = dataset[:,0:leng]
        ever = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=[13],dtype=numpy.str)
        ever[ever=="b'E'"]=numpy.array([1])
        ever[ever=="b's'"]=numpy.array([0])
        Y=ever
    else:
        dataset = numpy.loadtxt(inFile,skiprows=1,delimiter="|",usecols=range(1,13))
        leng=dataset.shape[1]
        X = dataset[:,0:leng]
        Y=None
    return X,Y,label

def SplitArray(array,makeEval):
    if makeEval:
        X=numpy.array([array[0:len(array)-1]])
        Y=numpy.array([array[len(array)-1]])
        ever[ever=="b'E'"]=numpy.array([1])
        ever[ever=="b's'"]=numpy.array([0])
        Y=ever
    else:
        X=numpy.array([array])
        Y=None
    return X,Y

def MakePred(loaded_model,X,Y,csvscores,makeEval,i):
    predictions = loaded_model.predict(X,verbose=1)
    to_write=numpy.around(predictions,decimals=0,out=None)
    if makeEval:
        print(type(numpy.array([Y[i]])))
        scores = loaded_model.evaluate(X,numpy.array([Y[i]]),verbose=0)
        scores_perc=scores[1]*100
        csvscores.append(scores_perc)
    return to_write,csvscores

def PrintPred(fo,label,i,to_write,makeEval,Y,x):
    lab=(str(label[i])).split("'")[1]
    #pred=int(to_write[0])
    trend=','.join(str(a) for a in x)
    if to_write[0]==1:
        pred='E'
    else:
        pred='s'
    # pred=to_write[0]
    # str(pred[pred==1])='E'
    # pred[pred==0]='s'
    #real=int(Y[i])
    if int(Y[i])==1:
        real='E'
    else:
        real='s'
    # real[pred==1]='E'
    # real[pred==0]='s'
    if not makeEval:
        fo.write(lab+','+trend+','+str(pred)+"\n")
    else:
        fo.write(lab+','+trend+','+str(pred)+','+str(real)+"\n")

def PrintEval(outFile,csvscores):
    with open(outFile+'_evaluation', "w") as fo:
        logging.info("RESULTS FOR EVALUATION: %.2f%% (+/- %.2f%%)"%(numpy.mean(csvscores), numpy.std(csvscores)))
        fo.write('Results for evaluate\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(csvscores), numpy.std(csvscores)))


def main(argv):
    # load
    start=datetime.now()
    logging.info("START")
    args=argparser.parse_args()
    inFile=args.infile
    outFile=args.outfile
    modelWeight=args.weights
    modelJson=args.json
    makeEval=args.eval
    #logging.info("LOADING THE DATA SET")
#    print(inFile)
    #dataset = numpy.loadtxt(inFile, delimiter=",")
    loaded_model=LoadModel(modelJson,modelWeight)
    file_user=open(inFile,'r')
    fo=open(outFile, "w")
    fo.write("kwd,pred,real\n")
    print("file created")
    X,Y,label=LoadDataset(inFile,makeEval)
    #print("DIM DATASET",type(dataset))
    print("Loop start")
    csvscores = []
    #for i,x in numpy.ndenumerate(X):
    #    print(X)
    for i, x in enumerate(X):
        print(x)
        #X,Y=SplitArray(array,makeEval)
        to_write,csvscores=MakePred(loaded_model,numpy.array([x]),Y,csvscores,makeEval,i)
        PrintPred(fo,label,i,to_write,makeEval,Y,x)
#        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    #USARE QUESTI PER LA SCRITTURA DEL FILE CON LE PREVISION
    PrintEval(outFile,csvscores)
    fo.close()
    file_user.close()
    logging.info("END")
    logging.info("DONE IN %d SEC"%(datetime.now()-start).total_seconds())




if __name__=="__main__":
    main(sys.argv)

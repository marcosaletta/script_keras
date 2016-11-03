# Feature Importance with Extra Trees Classifier
#from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
import numpy
import os
import sys
from getopt import getopt
import logging
import time
from sklearn.cross_validation import StratifiedKFold
import random
from sklearn.externals import joblib

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seed = 7
numpy.random.seed(seed)

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
   print("Usage: %s -t number of tree" % sys.argv[0])
   print("Usage: %s -k number of folds for cross validation" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/run_2/List4Keras_ALL_10k_train_69_list_train_r2 -o /home/marco/working-dir/Keras/results/run_3_RandClass/RandClass_10k_train_69_list_train -t 500 -k 10  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,numFold,numTree):
    start_time = time.clock()
    logging.info("START")
    logging.info("LOADING THE DATA SET")
#    print(inFile)
    dataset = numpy.loadtxt(inFile, delimiter=",",skiprows=1)
    # split into input (X) and output (Y) variables
    len=dataset.shape[1]
    X = dataset[:,0:len-1]
    Y = dataset[:,len-1]
    logging.info("SPLITTING SAMPLE FOR CROSS-VALIDATION WITH %i FOLD"%numFold)
    kfold = StratifiedKFold(y=Y,n_folds=numFold, shuffle=True, random_state=seed)
    cvscores = []
    logging.info("STARTING WITH CROSS-VALIDATION")
    epoch_num=0
    cvscores = []
    best_results=0
    best_fold=-1
#    print('KFOLD_eval:',enumerate(kfold))
#    for i, (train, test) in enumerate(kfold):
    for i, (train, test) in enumerate(kfold):
        logging.info("-------FOLD N. %i ---------"%i)
        logging.info("PREPARING MODEL")
        model = RandomForestClassifier(n_jobs=1,max_features=2,n_estimators=numTree)
        model.fit(X[train], Y[train])
        results=model.score(X[test],Y[test])
        logging.info("RESULTS FOR FOLD %i: %.2f%%"%(i,results*100))
        cvscores.append(results* 100)
        if best_results < results:
            model2save=model
            best_fold=i
    logging.info("TOTAL RESULTS")
    logging.info("MEAN AND STANDARD DEVIATION")
    logging.info('RESULTS FOR EVALUATE')
    print("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))
#    logging.info('RESULTS FOR MY TEST')
#    print("%.2f%% (+/- %.2f%%)" )% (numpy.mean(csvpred), numpy.std(csvpred))
    print(time.clock() - start_time, "seconds")
    with open(outFile, "w") as fo:
        fo.write('Results for evaluate\n')
        fo.write("%.2f%% (+/- %.2f%%)\n"% (numpy.mean(cvscores), numpy.std(cvscores)))
        fo.write(str(time.clock()-start_time)+"seconds\n")
    logging.info("WRITING THE BEST MODEL (FOLD %i) TO FILE"%best_fold)
    joblib.dump(model2save, outFile+'_model.pkl')
    logging.info("END")

    print(time.clock() - start_time, "seconds")


if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"f:o:t:k:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-t' in opts:
        numTree=int(opts['-t'])
    if '-k' in opts:
        numFold=int(opts['-k'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-f' not in opts and '-o' not in opts and '-t' not in opts and '-k' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,numFold,numTree)

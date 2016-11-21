import numpy
import os
import sys
from getopt import getopt
import logging
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
import random
from sklearn.externals import joblib
import pickle
import time
import luigi

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



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
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################

def main(inFile,outFile,numFold):
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
    kfold = StratifiedKFold(y=Y, n_folds=numFold, shuffle=True, random_state=seed)
    cvscores = []
    logging.info("STARTING WITH CROSS-VALIDATION")
    epoch_num=0
    cvscores = []
    best_results=0
    best_fold=-1
    for i, (train, test) in enumerate(kfold):
        logging.info("-------FOLD N. %i ---------"%i)
        logging.info("PREPARING SVM")
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(X[train],Y[train])
        results=clf.score(X[test],Y[test])
        if best_results < results:
            model2save=clf
            best_fold=i
        logging.info("RESULTS FOR FOLD %i: %.2f%%"%(i,results*100))
        cvscores.append(results* 100)
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
    #joblib.dump(model2save, outFile+'_model.pkl')
    picluigi=luigi.LocalTarget(outFile+'_model_luigi.pickle',format=luigi.format.Nop).open('w')
    pickle.dump(model2save, picluigi)
    picluigi.close()
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
    # if '-e' in opts:
    #     numEpoch=int(opts['-e'])
    if '-h' in opts:
       usage('msg')
    if '-f' not in opts and '-o' not in opts and '-k' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,numFold)

# Feature Importance with Extra Trees Classifier
#from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy
import os
import sys
from getopt import getopt
import logging

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
   print("Usage: %s -s number of features in the array" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,setSize):
# load data
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(url, names=names)
#array = dataframe.values
    logging.info("START")
    logging.info("LOAGING FEATURES FROM %s"%inFile)
    logging.info("USING %i FEATURES"%setSize)
    array = numpy.loadtxt(inFile, delimiter=",")
    X = array[:,0:69]
    Y = array[:,69]
# feature extraction
    model = ExtraTreesClassifier()
    fit=model.fit(X, Y)
    features = fit.transform(X)
    print('--------------',features.shape)
    print(model.feature_importances_)
    sort=numpy.sort(model.feature_importances_)
    print(sort)
    print(numpy.argsort(model.feature_importances_))
    print(features[0:10,:])
    logging.info("END")


if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"f:o:s:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-s' in opts:
        setSize=int(opts['-s'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-f' not in opts and '-o' not in opts and '-s' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,setSize)

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
   print("Usage: %s -t number of tree" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_valid -o ../results/test_random_tree_r2 -s 278 -t 20  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,setSize,numTree,mapFile):
# load data
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(url, names=names)
#array = dataframe.values
    logging.info("START")
    logging.info("LOAGING FEATURES FROM %s"%inFile)
    logging.info("USING %i FEATURES"%setSize)
    logging.info("USING %i TREES"%numTree)
    with open(inFile,'r') as inp:
        header=inp.readline().split(',')
    array = numpy.loadtxt(inFile, delimiter=",",skiprows=1)
    len=array.shape[1]
    X = array[:,0:len-1]
    Y = array[:,len-1]
# feature extraction
    model = ExtraTreesClassifier(n_estimators=numTree)
    fit=model.fit(X, Y)
    features = fit.transform(X)
    print('>>>>>>>>>>>>>>>>>>>>>>',X.shape)
    print('--------------',features.shape)
    print(model.feature_importances_)
    sort=numpy.sort(model.feature_importances_)
    print(sort)
    print(numpy.argsort(model.feature_importances_))
    print(features.shape)
    print(features[0:5,:])
    print("----------------lllllllll",type(features))
    numpy.savetxt(outFile,features,fmt='%.i',delimiter=',',newline='\n')
    with open(outFile+'_description','w')as des, open(mapFile,'r') as mapp:
        maps=mapp.readlines()
        for line in maps:
            for item in header:
                if item in line:
                    des.write(line+'\n')


    logging.info("END")


if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"f:o:s:t:m:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-s' in opts:
        setSize=int(opts['-s'])
    if '-t' in opts:
        numTree=int(opts['-t'])
    if '-m' in opts:
        mapFile=str(opts['-m'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-f' not in opts and '-o' not in opts and '-s' not in opts and '-t' not in opt and '-m' not in opt and '-h' not in opts:
        usage('0')
    main(inFile,outFile,setSize,numTree,mapFile)

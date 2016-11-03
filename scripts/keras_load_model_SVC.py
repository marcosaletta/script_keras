import os
import sys
from getopt import getopt
import logging
from sklearn.cross_validation import StratifiedKFold
import numpy
import random
from datetime import datetime
import time
from sklearn.externals import joblib
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
   print("Usage: %s -m model file (model.pkl)" % sys.argv[0])
   print("Usage: %s -t json model type (SVC or other)" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,model,wSex,ModelType):

    # load
    logging.info("START")
    logging.info("LOADING THE DATA SET")
#    print(inFile)
    #dataset = numpy.loadtxt(inFile, delimiter=",")
    logging.info("LOADING MODEL")
        # load weights into new model
    logging.info("LOADING WEIGHTS IN NEW MODEL")
    loaded_model = joblib.load(model)
    logging.info("LOADED MODEL FROM FILE")
    #evaluating performance on the set
    file_user=open(inFile,'r')
    next(file_user)
    fo=open(outFile, "w")
    print("file created")
#    sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
#    fo.write("START:"+str(sttime))
    evaluate=True
    right=0
    wrong=0
    cvscores = []
    print("Loop start")
    pos=0
    for line in file_user:
        pos+=1
        print("SONO ALLA LINEA: ",pos)
        try:
            first_split=line.split("|")
            code=first_split[0]
            first_split=first_split[1]
        except IndexError:
            first_split=line
            code="NONE"
        try:
            array=numpy.fromstring(first_split, sep=',')
        except:
            continue

        len_array=len(array)
#        print(len_array)
    # split into input (X) and output (Y) variables
        if wSex==1:
            X = numpy.array([array[0:len_array-1]])
            Y = numpy.array([array[len_array-1]])
        else:
            X = numpy.array([array[0:len_array]])
            evaluate=False
        code=first_split[0]
#        print(X)
#        print('-------------',X)
        predictions = loaded_model.predict(X)
        if ModelType=="SVC":
            score=loaded_model.decision_function(X)
        else:
            score=loaded_model.predict_proba(X)
#        predictions = loaded_model.predict_on_batch(X)
#        to_write = [round(x) for x in predictions]
        to_write=str(numpy.around(predictions,decimals=0,out=None))
        if evaluate==True:
            print("Y=",Y)
            if ModelType=="SVC":
                results=loaded_model.score(X,Y)
            else:
                results=loaded_model.score(X,Y)
            cvscores.append(results* 100)
#            print(to_write)
            # if to_write==Y[0]:
            #     right+=1
            # else:
            #     wrong+=1
            fo.write(str(code)+','+(to_write[0])+str(numpy.around(score,decimals=3,out=None))+"\n")
#        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    #USARE QUESTI PER LA SCRITTURA DEL FILE CON LE PREVISION
        else:
            fo.write(str(code)+','+(to_write[0])+str(numpy.around(score,decimals=3,out=None))+"\n")
#    print(evaluate)
    if evaluate==True:
        logging.info("TOTAL RESULTS")
        logging.info("MEAN AND STANDARD DEVIATION")
        logging.info('RESULTS FOR EVALUATE')
        print("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))
        fo.write("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))
    sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
    fo.write("END:"+str(sttime))
    fo.close()
    file_user.close()




if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"m:j:f:o:s:t:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-m' in opts:
        model=str(opts['-m'])
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-s' in opts:
        wSex=int(opts['-s'])
    if '-s' in opts:
        ModelType=str(opts['-t'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-f' not in opts and '-o' not in opts and '-m' not in opts and '-s' not in opts and '-t' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,model,wSex,ModelType)

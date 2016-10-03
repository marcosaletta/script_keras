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
from datetime import datetime

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
   print("Usage: %s -w model file (model.h5)" % sys.argv[0])
   print("Usage: %s -j json model file (model.json)" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



def main(inFile,outFile,modelWeight,modelJson):

    # load
    logging.info("START")
    logging.info("LOADING THE DATA SET")
#    print(inFile)
    #dataset = numpy.loadtxt(inFile, delimiter=",")
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
    file_user=open(inFile,'r')
    fo=open(outFile, "w")
    sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
    fo.write(sttime,"START")
    evaluate=True
    right=0
    wrong=0
    for line in file_user:
        array=numpy.fromstring(line, sep=',')
    # split into input (X) and output (Y) variables
        if len(array)==71:
            X = numpy.array([array[1:70]])
            Y = numpy.array([array[70]])
        else:
            X = numpy.array([array[1:71]])
        code=array[0]
        evaluate=False
        predictions = loaded_model.predict(X,verbose=1)
#        predictions = loaded_model.predict_on_batch(X)
#        to_write = [round(x) for x in predictions]
        to_write=numpy.around(predictions,decimals=0,out=None)
        if evaluate==True:
            print(Y)
            print(to_write)
            if to_write==Y[0]:
                right+=1
            else:
                wrong+=1
#        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    #USARE QUESTI PER LA SCRITTURA DEL FILE CON LE PREVISION
        else:
            fo.write(str(code,',',to_write[0])+"\n")
    if evaluate==True:
        perc_right=float(right)/float(right+wrong)
        fo.write("Right:"+str(right)+"\nWrong:"+str(wrong)+"\nPercentual:"+str(perc_right))
    sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
    fo.write(sttime,"END")
    fo.close()
    file_user.close()




if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"w:j:f:o:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-w' in opts:
        modelWeight=str(opts['-w'])
    if '-j' in opts:
        modelJson=str(opts['-j'])
    if '-f' in opts:
        inFile=str(opts['-f'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-f' not in opts and '-o' not in opts and '-w' not in opts and '-j' not in opts and '-h' not in opts:
        usage('0')
    main(inFile,outFile,modelWeight,modelJson)

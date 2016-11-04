import os
import sys
from getopt import getopt
import logging
import collections
import time
import fileinput
import pprint




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
   print("Usage: %s -r file with pred for RandomForest " % sys.argv[0])
   print("Usage: %s -n file with pred for NN" % sys.argv[0])
   print("Usage: %s -s file with pred for SVC" % sys.argv[0])
   print("Usage: %s -o output file" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################


def SplitLine(line):
#        print("SONO ALLA LINEA: ",pos)
    skipp=0
    try:
        line_split=line.split(",")
        code=line_split[0]
        pred=line_split[1]
    except IndexError:
        pred="NONE"
        code="NONE"
    return code, pred



def CreateList(rf,nn,svm):
    codes=[]
    preds=[]
    #line=item.readline()
    line_rf=next(rf,"FINE")
    line_nn=nn.readline()
    line_svm=svm.readline()
    list_lines=[line_rf, line_nn, line_svm]
    for item in list_lines:
        code, pred =SplitLine(item)
        codes.append(code)
        preds.append(pred.strip('[]').strip('.]\n').strip())
    print("------------------------",line_rf)
    return line_rf, codes, preds



def PrintResult(codes,preds,out,err):
    counter_codes=collections.Counter(codes)
    counter_preds=collections.Counter(preds)
    if len(counter_codes)>1:
        print('error code')
        code="NONE"
        err.write(pprint.pformat(counter_codes)+'\n')
    else:
        for key in counter_codes.keys():
            code=key
    if counter_preds['1']>counter_preds['0']:
#        sex='F'
        sex=1
        agr=counter_preds['1']
    elif counter_preds['1']<counter_preds['0']:
#        sex='M'
        sex=0
        agr=counter_preds['0']
    else:
        sex="NONE"
    if sex!="NONE":
        out.write(code+','+str(sex)+','+str(agr)+'\n')
        print("SESSO:",sex,"AGR:",str(agr))
    else:
        err.write(code+','+str(sex)+'\n')

    print("Counter:",counter_preds)
    print(preds)
#    print("SESSO:",sex,"AGR:",str(agr))





def main(PredRF,PredNN,PredSVM,outFile):
    start_time = time.clock()
    logging.info("START")
    logging.info("PREPARING ALL THE FILES")
    with open(PredRF,'r') as rf, open(PredNN,'r') as nn, open(PredSVM,'r') as svm, open(outFile,'w') as out, open(outFile+'_erors','w') as err:
#        line_rf=rf.readline()
        line_rf="INIZIO"
        logging.info("THE EOF FILE IS BASED ON THE RF FILE")
        logging.info("READ FIRST LINE FOR RF")
        while line_rf!="FINE":
            line_rf, codes, preds = CreateList(rf,nn,svm)
            if line_rf!="FINE":
                PrintResult(codes,preds,out,err)
            #print("linerf",line_rf)
    logging.info("END")












if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"r:n:s:o:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-r' in opts:
        PredRF=str(opts['-r'])
    if '-n' in opts:
        PredNN=str(opts['-n'])
    if '-s' in opts:
        PredSVM=str(opts['-s'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-r' not in opts and '-o' not in opts and '-n' not in opts and '-s' not in opts and '-h' not in opts:
        usage('0')
    main(PredRF,PredNN,PredSVM,outFile)

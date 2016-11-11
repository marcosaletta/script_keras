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
import luigi
import pickle as pickle
from luigi.format import UTF8
from luigi.format import FileWrapper
from os import listdir
from os.path import isfile, join

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("luigi").setLevel(logging.WARNING)


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


class PrepareLine(luigi.Task):
    """class to prepare the line for the analisys"""
#    def __init__(self, line, wSex):
    line = luigi.Parameter()
    wSex= luigi.Parameter()


    def SplitLine(self):
#        print("SONO ALLA LINEA: ",pos)
        #print('<-------------->',self.line)
        skipp=0
        try:
            first_split=self.line.split("|")
            self.code=first_split[0]
            first_split=first_split[1]
        except IndexError:
            first_split=self.line
            self.code="NONE"
        try:
            self.array=numpy.fromstring(first_split, sep=',')
            skipp=0
            return skipp,self.array
        except:
            skipp=-1
            return skipp,self.array


    def NumpyArray(self,array):
        #print('wSex',type(self.wSex))
        self.array=array
        len_array=len(self.array)
        if self.wSex==str(1):
            X = numpy.array([self.array[0:len_array-1]])
            Y = numpy.array([self.array[len_array-1]])
        else:
            X = numpy.array([array[0:len_array]])
            Y=None
#            evaluate=False
        #print('Y after split',Y)
        return X,Y,self.code



class Analize(luigi.Task):
    """Class to make the actual analisys"""
    #def __init__(self, X, Y, cvscores, loaded_model, ModelType, wSex):
    X = luigi.Parameter()
    Y = luigi.Parameter()
    cvscores = luigi.Parameter()
    loaded_model = luigi.Parameter()
    ModelType = luigi.Parameter()
    wSex = luigi.Parameter()

    def MakePred(self):
        predictions = self.loaded_model.predict(self.X)
        if self.ModelType=="SVC":
            score=self.loaded_model.decision_function(self.X)
            score=str(int(numpy.around(score,out=None)))
        else:
            score=self.loaded_model.predict_proba(self.X)
            score=str((numpy.amax(numpy.around(score,decimals=3,out=None))))
        if self.wSex==1:
            results=self.loaded_model.score(self.X,self.Y)
            self.cvscores.append(results* 100)
            return predictions, score, self.cvscores
        else:
            return predictions, score

class PrintResult(luigi.Task):
    """Class to print analisys results"""
    #def __init__(self, tupPred, fo, code):
    tupPred = luigi.Parameter()
    fo = luigi.Parameter()
    code = luigi.Parameter()
    fo_eval =luigi.Parameter()

    def PrintPred(self):
        predictions=self.tupPred[0]
        score=self.tupPred[1]
        to_write=(numpy.around(predictions,out=None))
        self.fo.write(str(self.code)+','+str(int(to_write[0]))+','+score+"\n")

    def PrintEval(self,fo_eval):
        logging.info("TOTAL RESULTS")
        logging.info("MEAN AND STANDARD DEVIATION")
        logging.info('RESULTS FOR EVALUATE')
        cvscores=self.tupPred[2]
        print("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))
        self.fo_eval.write("%.2f%% (+/- %.2f%%) \n"% (numpy.mean(cvscores), numpy.std(cvscores)))









class TestTask(luigi.Task):
#    file_number = luigi.Parameter()
    inFile = luigi.Parameter()
    outFile = luigi.Parameter()
    model = luigi.Parameter()
    wSex = luigi.Parameter()
    ModelType = luigi.Parameter()

    def input(self):
#        return luigi.LocalTarget(self.inFile), luigi.LocalTarget(self.model)
        return luigi.LocalTarget(self.inFile)

    def input_model(self):
        logging.info("FILENAME '%s'"%self.model)
        return luigi.LocalTarget(self.model,format=luigi.format.Nop)
        #return luigi.LocalTarget(self.model,format=luigi.format.Nop)
#        return luigi.util.task_wraps(self.model)

    def output(self):
        return luigi.LocalTarget(self.outFile), luigi.LocalTarget(self.outFile+'_eval')

    # def run(self):
    #     with self.output().open('w') as out_file:
    #         with self.input().open('r') as in_file:
    #             for line in in_file:
    #                 for out in json_mapper.map_line(line, THRESHOLD):
    #                     out_file.write(out)

#    def requires(self): return run()

    def run(self):
        start_time = time.clock()
        logging.info("START")
        logging.info("LOADING THE DATA SET")
        logging.info("LOADING MODEL")
        logging.info("LOADING WEIGHTS IN NEW MODEL")
        mod=self.input_model()
        loaded_model = pickle.load(mod.open('r'))
        logging.info("LOADED MODEL FROM FILE")
        logging.info("inp:%s"%(type(mod.open('r'))))
        #loaded_model = joblib.load(self.input_model().open('r'))
#        file_user=open(inp,'r')
        logging.info("LOADING FILE IN")
        file_user=self.input().open('r')
        logging.info('LOADED FILE IN')
#        logging.info("inp:%s"%(mod.open('r').readline()))
        next(file_user)
        #fo=open(self.output(), "w")
        logging.info("LOADING FILE OUT")
        fo_l, fo_eval_l=self.output()
        fo=fo_l.open('w')
        fo_eval=fo_eval_l.open('w')
        logging.info("LOADED FILE OUT")
        #fo_eval=open(self.output()+'_eval','w')
        print("file created")
        #print >> "TEST"
        sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
    #    fo.write("START:"+str(sttime))
        cvscores = []
        print("Loop start")
        pos=0
        for line in file_user:
            pos+=1
            #print(line)
            line_proc=PrepareLine(line,self.wSex)
            skipp,array=line_proc.SplitLine()
            if skipp==-1:
                continue
            X,Y,code = line_proc.NumpyArray(array)
            #print(X)
            #print(Y)
            Pred=Analize(X, Y, cvscores, loaded_model, self.ModelType, self.wSex)
            tupPred=Pred.MakePred()
            ToPrint=PrintResult(tupPred, fo, code, fo_eval)
            ToPrint.PrintPred()
        if len(tupPred)==3:
            logging.info("PRINTING EVALUATION")
            ToPrint.PrintEval()
            fo_eval.write("task completed in %.2f seconds"%(time.clock() - start_time))
    #    sttime = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
    #    fo_eval.write("END:"+str(sttime))
        logging.info('EXECUTED IN %.2f SECONDS'%(time.clock() - start_time))
        fo.close()
        fo_eval.close()
        file_user.close()



class LotsOTasks(luigi.WrapperTask):

    def requires(self):
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for k in range(4):
            yield TestTask()


if __name__ == '__main__':
    luigi.run()


# if __name__=="__main__":
#     opts, args=getopt(sys.argv[1:],"m:j:f:o:s:t:h")
#     opts=dict(opts)
#     inFile=None
#     outFile=None
#     if '-m' in opts:
#         model=str(opts['-m'])
#     if '-f' in opts:
#         inFile=str(opts['-f'])
#     if '-o' in opts:
#         outFile=str(opts['-o'])
#     if '-s' in opts:
#         wSex=int(opts['-s'])
#     if '-s' in opts:
#         ModelType=str(opts['-t'])
#     if '-h' in opts:
#         usage('msg')
# #    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
#     if '-f' not in opts and '-o' not in opts and '-m' not in opts and '-s' not in opts and '-t' not in opts and '-h' not in opts:
#         usage('0')
#     main(inFile,outFile,model,wSex,ModelType)

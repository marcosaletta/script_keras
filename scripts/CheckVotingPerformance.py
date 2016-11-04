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
   print("Usage: %s -v file with pred voted" % sys.argv[0])
   print("Usage: %s -s file with actual sex" % sys.argv[0])
   print("Usage: %s -o output file" % sys.argv[0])
   print("Usage: %s -h help\n" % sys.argv[0])
   print("Example:python %s -f /home/marco/working-dir/Krux/anagrafica_files/v2/List4Keras_ALL_aud_200916_FEW_FEATURES_valid_10K4test_loadModel -o ../results/test_load_model_small_set -w ../models/model.h5_small_set -j ../models/model.json_small_set  \n"%sys.argv[0])
   raise SystemExit


#####################################################################



class LineSplit:
    """Class to split line and retrive code and sex"""
    def __init__(self, line, type_file):
        self.line = line
        self.type_file = type_file

    def Split(self):
        splitted=self.line.split(',')
        print(self.type_file+"->"+str(type(splitted[0])))
        dict_line={}
        try:
            if splitted[0]==str(1) or splitted[0]==str(0):
                dict_line['code']="NONE"
            else:
                dict_line['code']=splitted[0]
            if self.type_file=='sex':
                dict_line['sex']=splitted[-1].strip('\n')
            else:
                dict_line['sex']=splitted[1].strip('\n')
        except:
            dict_line={}
        return dict_line

class MakeCheck:
    """Class to check the accuracy of predictions after voting"""
    def __init__(self, dict_vot, dict_sex, right, wrong, errors):
        self.dict_sex = dict_sex
        self.dict_vot = dict_vot
        self.right=right
        self.wrong=wrong
        self.errors=errors

    def Check(self):
        print("sex",self.dict_sex['code'])
        print("vot",self.dict_vot['code'])
        if self.dict_sex['code']==self.dict_vot['code']:
            if self.dict_sex['sex']==self.dict_vot['sex']:
                self.right+=1
            else:
                self.wrong+=1
        else:
            self.errors+=1
        return self.right, self.wrong, self.errors


class PrintResults:
    """Class to print results"""

    def __init__(self, right, wrong, errors, out):
        self.right=right
        self.wrong=wrong
        self.errors=errors
        self.out=out

    def Print(self):
        lines=self.right+self.wrong+self.errors
        per_right=self.right*100/lines
        per_wrong=self.wrong*100/lines
        per_err=self.errors*100/lines
        self.out.write("Right: "+str(per_right)+"Wrong: "+str(per_wrong)+"Errors: "+str(per_err)+'\n')
        return lines, per_right, per_wrong, per_err





def main(FileVoting,FileSex,outFile):
    start_time = time.clock()
    logging.info("START")
    right=0
    wrong=0
    errors=0
    with open(FileVoting,'r') as vot, open(FileSex,'r') as sex, open(outFile,'w') as out:
        line_sex="INIZIO"
        while line_sex!="FINE":
            line_sex=next(sex,'FINE')
            if line_sex!="FINE":
                line_vot=vot.readline()
                ob_sex=LineSplit(line_sex,'sex')
                dict_sex=ob_sex.Split()
                ob_vot=LineSplit(line_vot,'vot')
                dict_vot=ob_vot.Split()
                if len(dict_vot)==0 or len(dict_sex)==0:
                    print("EMPTY DICT, SKIPP")
                    continue
                ob_check=MakeCheck(dict_vot, dict_sex, right, wrong, errors)
                right, wrong, errors = ob_check.Check()
            else:
                continue
        ob_print=PrintResults(right, wrong, errors, out)
        lines, per_right, per_wrong, per_err=ob_print.Print()
        logging.info("ANALIZED %i LINES in %f"%(lines,(time.clock() - start_time)))
        logging.info("PERC RIGHT: %f WRONG: %f ERRORS: %f"%(per_right,per_wrong,per_err))
        logging.info("END")




if __name__=="__main__":
    opts, args=getopt(sys.argv[1:],"v:s:o:h")
    opts=dict(opts)
    inFile=None
    outFile=None
    if '-v' in opts:
        FileVoting=str(opts['-v'])
    if '-s' in opts:
        FileSex=str(opts['-s'])
    if '-o' in opts:
        outFile=str(opts['-o'])
    if '-h' in opts:
        usage('msg')
#    if '-f' not in opts==True and '-o' not in opts==True and '-m' not in opts==True and '-j' not in opts==True and '-h' not in opts==True:
    if '-v' not in opts and '-o' not in opts and '-s' not in opts and '-h' not in opts:
        usage('0')
    main(FileVoting,FileSex,outFile)

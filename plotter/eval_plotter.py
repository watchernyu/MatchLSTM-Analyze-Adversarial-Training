# use this file to do plotting on test results, (not on training results)
import os
import pprint
import matplotlib.pyplot as plt
from plotter_helpers import *

pp = pprint.PrettyPrinter()

# specify folder name and file names here for plotting
plot_folder_name = "../final_results_from_hpc/final/testing"
filenames = ['oritest.tsv','endtest.tsv','fronttest.tsv','any1test.tsv','any2test.tsv','any3test.tsv']
n_model = len(filenames)

losses = get_array(n_model,4)
f1s = get_array(n_model,4)
ems = get_array(n_model,4)
model_names = []

def read_testplot_file(filepath, modelIndex):
    fpt = open(filepath, "r")
    model_name = fpt.readline().strip()
    model_names.append(model_name)
    _ = fpt.readline()
    lines = fpt.readlines() # each line is in the format of: testset_name, loss, f1, em
    for i in range(len(lines)):
        tokens = lines[i].strip().split("\t")
        for j in range(1,len(tokens)):
            tokens[j] = float(tokens[j])

        lines[i] = tokens

    for c in range(4):
        losses[modelIndex][c] = lines[c][1]
        f1s[modelIndex][c] = lines[c][2]
        ems[modelIndex][c]  = lines[c][3]

for i in range(len(filenames)):
    filepath = os.path.join(plot_folder_name,filenames[i])
    read_testplot_file(filepath,i)

pp.pprint(losses)


# I'm going to write a function here to convert data into latex TABLE code !!!!!!!!!!!!
def toLatexTable(data,modelnames):
    s = ""
    s += "& Original & AddAny1 & AddOneSent & AddBestSent \\\\ \\hline\n"
    for r in range(len(modelnames)):
        s += modelnames[r]
        for num in data[r]:
            formated_num = "%0.2f" % num
            s += " & "+formated_num
        s+= "\\\\ \\hline\n"
    print s
    print ""

# just call these functions and then copy paste into the latex file you use
print "losses"
toLatexTable(losses,model_names)
print "F1"

toLatexTable(f1s,model_names)
print "Em"

toLatexTable(ems,model_names)
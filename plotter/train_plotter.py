# use this file to do plotting
# the job of this file is to read the contents of the specified plotlog files
# and then do some nice plots!
import os
import pprint
import matplotlib.pyplot as plt
from plotter_helpers import *

SAVETOFILE = False # set this to true when want to save to file
# set this to false when want to just view
save_name = "training_figure.png"
N_EPOCH = 12 # max number of epoch for plotting

pp = pprint.PrettyPrinter()

filenames = ["original_1210235153_plot.tsv","any_1_1210235209_plot.tsv","any_2_1210235209_plot.tsv","any_3_1210235210_plot.tsv"]

plotdata = []
losses = []
f1s = []
ems = []
model_names = []

def read_trainplot_file(filepath):
    fpt = open(filepath, "r")
    model_name = fpt.readline().strip()
    model_names.append(model_name)
    _ = fpt.readline()
    lines = fpt.readlines()
    for i in range(len(lines)):
        tokens = lines[i].strip().split("\t")
        for j in range(len(tokens)):
            tokens[j] = float(tokens[j])
        tokens[0] = int(tokens[0])

        lines[i] = tokens

    plotdata.append(lines)
    loss = get_colomn(lines,1)
    f1 = get_colomn(lines,2)
    em = get_colomn(lines,3)

    losses.append(loss)
    f1s.append(f1)
    ems.append(em)


for fn in filenames:
    filepath = os.path.join(plot_folder_name,fn)
    read_trainplot_file(filepath)

# model_display_names = []


epoches = [i for i in range(1, N_EPOCH + 1)] #used as x-axis
print epoches
print losses[0]


for i in range(len(losses)):
    plt.plot(epoches, losses[i][:N_EPOCH])


plt.legend(model_names)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)

if SAVETOFILE:
    plt.savefig(save_name)
else:
    plt.show()


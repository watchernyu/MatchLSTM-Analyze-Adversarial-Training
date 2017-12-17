# use this file to do plotting
# the job of this file is to read the contents of the specified plotlog files
# and then do some nice plots!
import os
import pprint
import matplotlib.pyplot as plt
from plotter_helpers import *

plot_folder_name = "../final_results_from_hpc/final/training"
filenames = ['oritrain.tsv','endtrain.tsv','fronttrain.tsv','any1train.tsv','any2train.tsv','any3train.tsv']
colors = ['g','y','r','c','b','#000000']

SAVETOFILE = False # set this to true when want to save to file
# set this to false when want to just view
save_name = "training_figure"
N_EPOCH = 24 # max number of epoch for plotting

pp = pprint.PrettyPrinter()


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

##############validation losses
for i in range(len(losses)):
    plt.plot(epoches, losses[i][:N_EPOCH],colors[i])

plt.legend(model_names)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)

if SAVETOFILE:
    plt.savefig(save_name+"_loss.png")
else:
    plt.show()

##############f1
for i in range(len(losses)):
    plt.plot(epoches, f1s[i][:N_EPOCH],colors[i])

plt.legend(model_names)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('f1 score', fontsize=14)

if SAVETOFILE:
    plt.savefig(save_name+"_f1.png")
else:
    plt.show()


##############em
for i in range(len(losses)):
    plt.plot(epoches, ems[i][:N_EPOCH],colors[i])

plt.legend(model_names)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('em score', fontsize=14)

if SAVETOFILE:
    plt.savefig(save_name+"_em.png")
else:
    plt.show()

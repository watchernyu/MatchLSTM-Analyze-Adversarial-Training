# Using match-lstm and SQuAD dataset to test how different settings of adversarial training affect model robustness
--------------------------------------------------------------------------------
The model implementation of this project is a work by xingdi-eric-yuan. We have built our code on top of his MatchLSTM implementation.
For the original MatchLSTM project, check out https://github.com/xingdi-eric-yuan/MatchLSTM-PyTorch

In this project we explore how adversarially-modified training data might be used to train machine reading comprehension models to bocome more robust.

## Environment
This project uses python 2.7 and pytorch 0.2
Other library dependency are in requirements.txt

## Generating Adversarial Examples (Random words)

The adversarial data can be generated using the generator in adv_gen folder, the new data are generated on top of SQuAD dataset original data sets, you can specify the paths, what data to generate with and how many random sequence to put into the data. You can also change the parameter in the program to indicate how long a sequence, how many common words/question words should be used in each random sequence.

## Training

Use train_new.py to train the model, use flags to indicate how the model should be trained.

Use train_new.py -h to find out all the possible flags

For example, use the following command to train the model with model_name as "original", to get data from new_advdata/original folder, generate a h5 data file called "original.h5", train 15 epoches and use a batch size of 96.

python train_new_debug.py -name original -d new_advdata/original -h5 original.h5 -fep 15 -fbs 96

## Generate error analysis files

Use the following command to do a error analysis evalution of the model. Make sure you use the same .h5 file, since it contains all the train, validation and test data that the model uses, along with unique word2index dictionaries that are generated from that data.

python train_new.py -name original -d new_advdata/original -h5 original.h5 -eonly -errana

## Plotting

You can use the files in plotter folder to do plotting. train_plotter.py gives a matplotlib graph for training traces. eval_plotter.py prints out latex code that you can plug into your latex file between begin and end of a table and will give you a table for test time results.

# Using match-lstm and SQuAD dataset to test how different settings of adversarial training affect model robustness
--------------------------------------------------------------------------------
The model implementation of this project is a work by xingdi-eric-yuan. We have built our code on top of his MatchLSTM implementation.
For the original MatchLSTM project, check out https://github.com/xingdi-eric-yuan/MatchLSTM-PyTorch

In this project we explore how adversarially-modified training data might be used to train machine reading comprehension models to bocome more robust.

The adversarial data can be generated using the generator in adv_gen folder, the new data are generated on top of SQuAD dataset original data sets.

Use train_new.py to train the model, use flags to indicate how the model should be trained.

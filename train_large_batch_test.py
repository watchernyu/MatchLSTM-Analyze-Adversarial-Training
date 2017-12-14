import os
import time
import h5py
import sys
import logging
import argparse
import yaml
import torch
from tqdm import tqdm
from lib.dataset.squad_dataset import SquadDataset
from lib.dataset.squad_dataset_testpart import SquadTestDataset # this is used to read in adv test sets
from lib.models.match_lstm import MatchLSTMModel
from lib.objectives.generic import StandardNLL
from lib.utils.setup_logger import setup_logging, log_git_commit
from helpers.generic import print_shape_info, print_data_samples, random_generator, squad_trim, add_char_level_stuff,\
    torch_model_summarize, generator_queue, evaluate

# this is the main training file, call train.py to train the model

logger = logging.getLogger(__name__)
wait_time = 0.01  # in seconds

DEFAULT_DATA_FOLDER_PATH = 'tokenized_squad_v1.1.2'
DEFAULT_CONFIG_FILENAME = 'config_mlstm.yaml'
DEFAULT_H5_FILENAME = 'squad_dataset.1.1.2.h5'
DEFAULT_MODEL_NAME = 'squad_original'
PLOTLOG_FOLDER = 'plotlogs'

def the_main_function(name_of_model,config_dir='config', update_dict=None,data_folder_path=DEFAULT_DATA_FOLDER_PATH,config_filename = DEFAULT_CONFIG_FILENAME,h5filename = DEFAULT_H5_FILENAME):
    # read config from yaml file

    print "data folder path: "+data_folder_path
    cfname = config_filename
    timestring = time.strftime("%m%d%H%M%S")

    # read configurations from config file
    config_file = os.path.join(config_dir, cfname)
    with open(config_file) as reader:
        model_config = yaml.safe_load(reader)

    # _n = - 1 means use all data
    _n = -1
    if args.t:
        # if doing tiny test
        # then only use 100 data
        # also, only run for 2 epoches
        _n = 100
        model_config['scheduling']['epoch'] = 15

    TRAIN_SIZE = _n
    VALID_SIZE = _n
    TEST_SIZE = _n

    # the model will be saved here, the name of the save file can be specified using -name
    model_save_path = os.path.join(model_config['dataset']['model_save_folder'],name_of_model)+".pt"

    # open plotlog file to do some extra logging for easy plotting later
    # this file records the validation result at end of each epoch
    plotlog_filename = name_of_model+"_"+timestring+"_plot.tsv"
    plotlog_path = os.path.join(PLOTLOG_FOLDER,plotlog_filename)
    plotlog_fpt = open(plotlog_path,'w')
    plotlog_fpt.write(name_of_model+"\n")
    plotlog_fpt.write("epoch\tnll_loss\tf1\tem\tlr\ttime\n")

    # this file is used to log the test set evaluation results
    testplotlog_filename = name_of_model+"_"+timestring+"_test.tsv"
    testplotlog_path = os.path.join(PLOTLOG_FOLDER,testplotlog_filename)
    testplotlog_fpt = open(testplotlog_path,'w')
    testplotlog_fpt.write(name_of_model+"\n")
    testplotlog_fpt.write("testset\tnll_loss\tf1\tem\n")

    # the dataset is basically all the things about squad data
    # dataset is built by calling the SquadDataset class
    # it takes the data in tokenized_squad_v1.1.2 directory
    # then convert them into a single .h5 file
    dataset = SquadDataset(dataset_h5=h5filename,
                           data_path=data_folder_path+"/",
                           ignore_case=True)

    # divide data into 3 parts
    train_data, valid_data, test_data = dataset.get_data(train_size=TRAIN_SIZE, valid_size=VALID_SIZE, test_size=TEST_SIZE)
    print_shape_info(train_data)
    if False:
        print('----------------------------------  printing out data shape')
        print_data_samples(dataset, train_data, 12, 15)
        exit(0)

    ##########################################################################################################
    # after read in the train, valid and dev set
    # train and valid have settings specified by folder name, dev set for each of 6 models is the original squad dataset
    # now for the following part, we get 3 extra testsets
    # add_any_4, add one sent and add best sent

    testdataset = SquadTestDataset(dataset_h5='squad_testset.1.0.h5',
                                   data_path='test_data' + "/",
                                   ignore_case=True)

    # after the model finished training, we will test its performance on these 3 datasets
    add_any_4_testdata, add_one_sent_testdata, add_best_sent_testdata = testdataset.get_data(train_size=TEST_SIZE,valid_size=TEST_SIZE, test_size=TEST_SIZE)  # each is a data dict
    ###########################################################################################################

    # Set the random seed manually for reproducibility.
    torch.manual_seed(model_config['scheduling']['cuda_seed'])
    if torch.cuda.is_available():
        if not model_config['scheduling']['enable_cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(model_config['scheduling']['cuda_seed'])
    else:
        print ("WARNING: no CUDA available, enable_cuda is disabled.")
        model_config['scheduling']['enable_cuda'] = False

    # here we add the option of continue training the model
    if args.r and os.path.isfile(model_save_path):
        logger.info('##### MODEL EXIST, CONTINUE TRAINING EXISTING MODEL #####')
        with open(model_save_path, 'rb') as save_f:
            _model = torch.load(save_f)
    else:
        logger.info('##### INIT UNTRAINED MODEL #####')
        # init untrained model
        _model = MatchLSTMModel(model_config=model_config, data_specs=dataset.meta_data)

    if model_config['scheduling']['enable_cuda']:
        _model.cuda()

    criterion = StandardNLL()
    if model_config['scheduling']['enable_cuda']:
        criterion = criterion.cuda()

    # print out a summarization of the model
    logger.info('finished loading models')
    logger.info(torch_model_summarize(_model))

    # get optimizer / lr
    init_learning_rate = model_config['optimizer']['learning_rate']
    parameters = filter(lambda p: p.requires_grad, _model.parameters())
    if model_config['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif model_config['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    input_keys = ['input_story', 'input_question', 'input_story_char', 'input_question_char']
    output_keys = ['answer_ranges']
    batch_size = 192 # set to 192
    valid_batch_size = model_config['scheduling']['valid_batch_size']
    _f = h5py.File(dataset.dataset_h5, 'r')
    word_vocab = _f['words_flatten'][0].split('\n')
    word_vocab = list(word_vocab)
    # word2id = dict(zip(word_vocab, range(len(word_vocab))))
    char_vocab = _f['words_flatten_char'][0].split('\n')
    char_vocab = list(char_vocab)
    char_word2id = {}
    for i, ch in enumerate(char_vocab):
        char_word2id[ch] = i

    # the generator uses yield to give batches of data, similar to what taught in class
    train_batch_generator = random_generator(data_dict=train_data, batch_size=batch_size,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=squad_trim, sort_by='input_story',
                                             char_level_func=add_char_level_stuff,
                                             word_id2word=word_vocab, char_word2id=char_word2id,
                                             enable_cuda=model_config['scheduling']['enable_cuda'])
    # train
    number_batch = (train_data['input_story'].shape[0] + batch_size - 1) // batch_size
    data_queue, _ = generator_queue(train_batch_generator, max_q_size=20)
    learning_rate = init_learning_rate
    best_val_f1 = None
    be_patient = 0

    starttime = time.time() # this is used to count how much time it takes to run a single epoch

    startEpoch = args.startEpoch # for example, if first run trained 12 epoches, then in the second run, use "-e 12" to tell the program
    # that you are running starting from epoch 12.
    try:
        for epoch in range(startEpoch,startEpoch+model_config['scheduling']['epoch']):
            _model.train()
            sum_loss = 0.0
            with tqdm(total=number_batch, leave=True, ncols=160, ascii=True) as pbar:
                for i in range(number_batch):
                    # qgen train one batch
                    generator_output = None
                    while True:
                        if not data_queue.empty():
                            generator_output = data_queue.get()
                            break
                        else:
                            time.sleep(wait_time)
                    input_story, input_question, input_story_char, input_question_char, answer_ranges = generator_output

                    optimizer.zero_grad()
                    _model.zero_grad()
                    preds = _model.forward(input_story, input_question, input_story_char, input_question_char)  # batch x time x 2
                    # loss
                    loss = criterion(preds, answer_ranges)
                    loss = torch.mean(loss)
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm(_model.parameters(), model_config['optimizer']['clip_grad_norm'])
                    optimizer.step()  # apply gradients
                    preds = torch.max(preds, 1)[1].cpu().data.numpy().squeeze()  # batch x 2
                    batch_loss = loss.cpu().data.numpy()
                    sum_loss += batch_loss * batch_size
                    pbar.set_description('epoch=%d, batch=%d, avg_loss=%.5f, batch_loss=%.5f, lr=%.6f' % (epoch, i, sum_loss / float(batch_size * (i + 1)), batch_loss, learning_rate))
                    pbar.update(1)

            # eval on valid set
            val_f1, val_em, val_nll_loss = evaluate(model=_model, data=valid_data, criterion=criterion,
                                                    trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                                    word_id2word=word_vocab, char_word2id=char_word2id,
                                                    batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
            logger.info("epoch=%d, valid nll loss=%.5f, valid f1=%.5f, valid em=%.5f, lr=%.6f, timespent=%d" % (epoch, val_nll_loss, val_f1, val_em, learning_rate,time.time()-starttime))

            # also log to plotlog file
            plotlog_fpt.write(str(epoch)+"\t"+str(val_nll_loss)+"\t"+str(val_f1)+"\t"+str(val_em)+"\t"+str(learning_rate)+"\t"+str(time.time()-starttime)+"\n")
            plotlog_fpt.flush()
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_f1 or val_f1 > best_val_f1:
                with open(model_save_path, 'wb') as save_f:
                    torch.save(_model, save_f)
                best_val_f1 = val_f1
                be_patient = 0
            else:
                if epoch >= model_config['optimizer']['learning_rate_decay_from_this_epoch']:
                    if be_patient >= model_config['optimizer']['learning_rate_decay_patience']:
                        if learning_rate * model_config['optimizer']['learning_rate_decay_ratio'] > model_config['optimizer']['learning_rate_cut_lowerbound'] * model_config['optimizer']['learning_rate']:
                            # Anneal the learning rate if no improvement has been seen in the validation dataset.
                            logger.info('cutting learning rate from %.5f to %.5f' % (learning_rate, learning_rate * model_config['optimizer']['learning_rate_decay_ratio']))
                            learning_rate *= model_config['optimizer']['learning_rate_decay_ratio']
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                        else:
                            logger.info('learning rate %.5f reached lower bound' % (learning_rate))
                    be_patient += 1

            # shouldn't look at test set
            # commenting out this part of the original implementation
            # test_f1, test_em, test_nll_loss = evaluate(model=_model, data=test_data, criterion=criterion,
            #                                            trim_function=squad_trim, char_level_func=add_char_level_stuff,
            #                                            word_id2word=word_vocab, char_word2id=char_word2id,
            #                                            batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
            # logger.info("test: nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))
            logger.info("========================================================================\n")

    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        logger.info('--------------------------------------------\n')
        logger.info('Exiting from training early\n')

    # Load the best saved model.
    with open(model_save_path, 'rb') as save_f:
        _model = torch.load(save_f)

    # Run on test data.
    logger.info("loading best model and evaluate on original squad test (dev) sets------------------------------------------------------------------\n")
    test_f1, test_em, test_nll_loss = evaluate(model=_model, data=test_data, criterion=criterion,
                                               trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                               word_id2word=word_vocab, char_word2id=char_word2id,
                                               batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
    logger.info("------------------------------------------------------------------------------------\n")
    logger.info("nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))
    testplotlog_fpt.write("OriginalSquad\t"+str(test_nll_loss) + "\t" + str(test_f1) + "\t" + str(test_em) + "\n")
    testplotlog_fpt.flush()

    logger.info("evaluate on add any 4 test set------------------------------------------------------------------\n")
    test_f1, test_em, test_nll_loss = evaluate(model=_model, data=add_any_4_testdata, criterion=criterion,
                                               trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                               word_id2word=word_vocab, char_word2id=char_word2id,
                                               batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
    logger.info("------------------------------------------------------------------------------------\n")
    logger.info("nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))
    testplotlog_fpt.write("AddAny4\t"+str(test_nll_loss) + "\t" + str(test_f1) + "\t" + str(test_em) + "\n")
    testplotlog_fpt.flush()
    logger.info("evaluate on add one sent test set------------------------------------------------------------------\n")
    test_f1, test_em, test_nll_loss = evaluate(model=_model, data=add_one_sent_testdata, criterion=criterion,
                                               trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                               word_id2word=word_vocab, char_word2id=char_word2id,
                                               batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
    logger.info("------------------------------------------------------------------------------------\n")
    logger.info("nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))
    testplotlog_fpt.write("AddOneSent\t"+str(test_nll_loss) + "\t" + str(test_f1) + "\t" + str(test_em) + "\n")
    testplotlog_fpt.flush()
    logger.info("evaluate on add best sent test set------------------------------------------------------------------\n")
    test_f1, test_em, test_nll_loss = evaluate(model=_model, data=add_best_sent_testdata, criterion=criterion,
                                               trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                               word_id2word=word_vocab, char_word2id=char_word2id,
                                               batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
    logger.info("------------------------------------------------------------------------------------\n")
    logger.info("nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))
    testplotlog_fpt.write("AddBestSent\t"+str(test_nll_loss) + "\t" + str(test_f1) + "\t" + str(test_em) + "\n")
    testplotlog_fpt.flush()
    return


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding("utf-8")

    # make a directory for logs and saved models if their directory don't exist yet.
    for _p in ['logs', 'saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO, add_time_stamp=True)
    # goal_prompt(logger)
    log_git_commit(logger)
    parser = argparse.ArgumentParser(description="train network.")
    # parser.add_argument("result_file_npy", help="result file with npy format.")  # position argument example.
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-t", action='store_true', help="tiny test, set this flag if you want to do some fast test, the model will only use 100 data entries for everything")
    parser.add_argument("-r", action='store_true',
                        help="continue to train an existing model, the program will check if the model exist, if so then load the model and continue to train it")

    parser.add_argument("-d","--datapath",help="specify path to training data",default=DEFAULT_DATA_FOLDER_PATH ,type=str)
    parser.add_argument("-h5","--datah5",help="specify filename of squad h5 file, you can simply sepecify a name related to the datapath",default=DEFAULT_H5_FILENAME ,type=str)

    parser.add_argument("-name","--nameOfModel",help="specify the name of the model to save",default=DEFAULT_MODEL_NAME ,type=str)
    parser.add_argument("-e", "--startEpoch", help="specify the starting epoch of the model, this is used when continue training existing model",
                        default=0, type=int)

    args = parser.parse_args()

    config_to_use = DEFAULT_CONFIG_FILENAME

    the_main_function(name_of_model=args.nameOfModel,config_dir=args.config_dir,data_folder_path=args.datapath,config_filename=config_to_use,h5filename=args.datah5)

# this program uses the common_1000 txt and the squad tokenized dataset to generate
# some adv training data
import os
import random
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

original_data_folder_path = "../tokenized_squad_v1.1.2"
train_question_filename = "train-v1.1-question.txt"
train_story_filename = "train-v1.1-story.txt"
train_answer_filename = "train-v1.1-answer-range.txt"


valid_question_filename = "valid-v1.1-question.txt"
valid_story_filename = "valid-v1.1-story.txt"
valid_answer_filename = "valid-v1.1-answer-range.txt"


test_question_filename = "dev-v1.1-question.txt"
test_story_filename = "dev-v1.1-story.txt"
test_answer_filename = "dev-v1.1-answer-range.txt"

new_data_folder_path = "../squad_adv_2" # the data generated will be put into this folder

N_ADV_WORDS = 10 # this is the number of random words that we put at the end of the sentence (or elsewhere)

INSERT_MODE_FRONT = "FRONT"
INSERT_MODE_END = "END"
INSERT_MODE_ANYPLACE = "ANYPLACE"

def get_1000_common():
    # this function will return a list,
    # the list contains the 1000 words
    common_1000_file = open("common_1000.txt","r")
    common_words = common_1000_file.read().split("\t")
    print common_words[:10]
    return common_words

COMMON_WORDS = get_1000_common()

def questionLineToTokens(line):
    # use this function to convert a single question (in the form of a line) to tokens, so it's easier to use
    # when we try to generate new word
    tokens = line.split(" ")
    return tokens[:-1]

def questionsToTokens(questions):
    # use this function to convert a list of questions to tokens (each question in the form of a line)
    tokenized_questions = []

    for i in range(len(questions)):
        tokenized_questions.append(questionLineToTokens(questions[i]))
    return tokenized_questions

# New Rule: Generate sequence of new words that contains at least some question words.
def sample_common(common, num_sample):
    # sample 20 words from common
    sampled_common = np.random.choice(common, num_sample)
    return sampled_common

def generate_sequence(sampled_common, question, num_question_words, num_common_words):
    # this function returns a sentence in the form of a list
    # the sentence has at least <num_question_words> words from question line
    # and at least <num_common_words> words from the common words
    common_num = random_num = random.randint(num_common_words, N_ADV_WORDS - num_question_words)
    common = list(np.random.choice(sampled_common, common_num))
    q = list(np.random.choice(question, N_ADV_WORDS - common_num))
    sequence = common + q 
    random.shuffle(sequence)
    return sequence + ['.']

def write_to_file(file,listOfLines):
    # this function will write a list of sentences to a txt file
    for line in listOfLines:
        file.write(line+"\n")

def generate_adversarial_data(original_data_folder_path,story_filename,question_filename,new_data_folder_path, answer_filename,insert_mode=INSERT_MODE_ANYPLACE,num_insertions=1):
    # this function should be run for training data and validation data
    if not os.path.exists(new_data_folder_path):
        os.mkdir(new_data_folder_path)

    story_path = os.path.join(original_data_folder_path, story_filename)
    question_path = os.path.join(original_data_folder_path, question_filename)
    answer_path = os.path.join(original_data_folder_path, answer_filename)

    train_story_file = open(story_path,"r")
    story_lines = train_story_file.read().strip().split("\n")

    train_question_file = open(question_path,"r")
    raw_question_lines = train_question_file.read().strip().split("\n")
    question_lines = questionsToTokens(raw_question_lines)

    train_answer_file = open(answer_path, "r")
    answer_lines = train_answer_file.read().strip().split("\n")

    print len(story_lines)
    print len(question_lines)
    print len(answer_lines)

    print story_lines[0]
    print question_lines[0]
    print answer_lines[0]

    adv_train_story_filepath = os.path.join(new_data_folder_path, story_filename)
    adv_train_story_file = open(adv_train_story_filepath,"w")

    adv_train_answer_filepath = os.path.join(new_data_folder_path, answer_filename)
    adv_train_answer_file = open(adv_train_answer_filepath, "w")

    adv_train_question_filepath = os.path.join(new_data_folder_path, question_filename)
    adv_train_question_file = open(adv_train_question_filepath, "w")

    n = len(story_lines)
    new_story_lines = []
    new_answer_lines = []
    new_question_lines = [] #we need new question lines because some questions (data entries) are corrupted for
    # some reason. Might be a problem of SQuAD, so in the new question lines, we delete those invalid questions
    for i in range(n): # for each line
        answer_list = answer_lines[i].strip().split(':')
        story_token_list = story_lines[i].strip().split()

        if len(question_lines[i]) > 1 and len(answer_list) == 2: # if this line is valid
            # if the question is valid, include it in the new question file
            new_question_lines.append(raw_question_lines[i])

            for j in range(num_insertions):
                sampled_common = sample_common(COMMON_WORDS, 20)
                sequence = generate_sequence(sampled_common, question_lines[i], 4, 3)

                insert_indices = [i for i, x in enumerate(story_token_list) if x == "."]
                insert_indices = [0] + insert_indices
                insert = np.random.choice(insert_indices,1)[0]

                #print ('insert',insert)
                # compare insert position with answer range to see if we need to change answer range
                answer_start = int(answer_list[0])
                answer_end = int(answer_list[1])
                #print ('answer_start',answer_start)


                if insert_mode==INSERT_MODE_FRONT:
                    insert = 0
                elif insert_mode == INSERT_MODE_END:
                    insert = len(story_token_list)-1

                # split the story into two parts
                if insert == 0: #insert at the beginning
                    first_half_story = []
                    second_half_story = story_token_list
                else: # insert anywhere
                    first_half_story = story_token_list[:insert + 1]
                    second_half_story = story_token_list[insert + 1:]

                if insert <= answer_start:
                    #print ('i',i)
                    answer_start += len(sequence)
                    answer_end += len(sequence)

                    #print ('new_answer_start', answer_start)
                new_story_list = first_half_story + sequence + second_half_story
                new_answer_list = [str(answer_start), str(answer_end)]

                if j == num_insertions - 1: # if this is the last insertion for this data entry
                    new_story_line = " ".join(new_story_list)
                    new_story_lines.append(new_story_line)
                    new_answer_line = ":".join(new_answer_list)
                    new_answer_lines.append(new_answer_line)
                else:
                    story_token_list = new_story_list
                    answer_list = new_answer_list
        else: # if this line is invalid, just skip it
            pass
            # print i
            # print question_lines[i]
            # print answer_lines[i]
        #except:
        #    pass
    # print story_lines[0]
    # print story_lines[1]
    # print story_lines[2]
    write_to_file(adv_train_story_file,new_story_lines)
    write_to_file(adv_train_answer_file,new_answer_lines)
    write_to_file(adv_train_question_file,new_question_lines)
    print "adversarial data generated at "+adv_train_story_filepath

def answerTokensToAnswerLine(answerTokens):
    # answerTokens take the form of e.g. [33,35,33,35,32,36]
    # the output is 33:35 ||| 33:35 ||| 32:36
    num_answers = int(len(answerTokens)/2)
    output_line = str(answerTokens[0])+":"+str(answerTokens[1])
    for i in range(1,num_answers):
        output_line+=" ||| "
        output_line+=(str(answerTokens[i*2])+":"+str(answerTokens[i*2+1]))

    return output_line

def generate_adversarial_test_data(original_data_folder_path,story_filename,question_filename,new_data_folder_path, answer_filename,insert_mode=INSERT_MODE_ANYPLACE,num_insertions=1):
    # this function should be run for training data and validation data
    # THIS IS A VERSION OF generate_adversarial_data that can deal with data with multiple answers
    # the data used in the original test set has multiple answers for each line, so we need to
    # make some modifications

    if not os.path.exists(new_data_folder_path):
        os.mkdir(new_data_folder_path)

    story_path = os.path.join(original_data_folder_path, story_filename)
    question_path = os.path.join(original_data_folder_path, question_filename)
    answer_path = os.path.join(original_data_folder_path, answer_filename)

    train_story_file = open(story_path,"r")
    story_lines = train_story_file.read().strip().split("\n")

    train_question_file = open(question_path,"r")
    raw_question_lines = train_question_file.read().strip().split("\n")
    question_lines = questionsToTokens(raw_question_lines)

    train_answer_file = open(answer_path, "r")
    answer_lines = train_answer_file.read().strip().split("\n")

    print len(story_lines)
    print len(question_lines)
    print len(answer_lines)

    print story_lines[0]
    print question_lines[0]
    print answer_lines[0]

    adv_train_story_filepath = os.path.join(new_data_folder_path, story_filename)
    adv_train_story_file = open(adv_train_story_filepath,"w")

    adv_train_answer_filepath = os.path.join(new_data_folder_path, answer_filename)
    adv_train_answer_file = open(adv_train_answer_filepath, "w")

    adv_train_question_filepath = os.path.join(new_data_folder_path, question_filename)
    adv_train_question_file = open(adv_train_question_filepath, "w")

    n = len(story_lines)
    new_story_lines = []
    new_answer_lines = []
    new_question_lines = [] #we need new question lines because some questions (data entries) are corrupted for
    # some reason. Might be a problem of SQuAD, so in the new question lines, we delete those invalid questions
    for i in range(n): # for each line
        answer_list = answer_lines[i].replace(" ","").replace("|||",":").strip().split(":")
        num_of_answers = int(len(answer_list)/2)
        story_token_list = story_lines[i].strip().split()

        if len(question_lines[i]) > 1 and len(answer_list) >= 2: # if this line is valid
            # if the question is valid, include it in the new question file
            new_question_lines.append(raw_question_lines[i])

            for j in range(num_insertions):
                sampled_common = sample_common(COMMON_WORDS, 20)
                sequence = generate_sequence(sampled_common, question_lines[i], 4, 3)

                insert_indices = [i for i, x in enumerate(story_token_list) if x == "."]
                insert_indices = [0] + insert_indices
                insert = np.random.choice(insert_indices,1)[0]



                if insert_mode==INSERT_MODE_FRONT:
                    insert = 0
                elif insert_mode == INSERT_MODE_END:
                    insert = len(story_token_list)-1

                # split the story into two parts
                if insert == 0: #insert at the beginning
                    first_half_story = []
                    second_half_story = story_token_list
                else: # insert anywhere
                    first_half_story = story_token_list[:insert + 1]
                    second_half_story = story_token_list[insert + 1:]

                new_answer_list = []
                for i_answer in range(num_of_answers):
                    answer_start = int(answer_list[i_answer*2])
                    answer_end = int(answer_list[i_answer*2+1])

                    if insert <= answer_start:
                        answer_start += len(sequence)
                        answer_end += len(sequence)

                    new_answer_list.append(str(answer_start))
                    new_answer_list.append(str(answer_end))

                new_story_list = first_half_story + sequence + second_half_story

                if j == num_insertions - 1: # if this is the last insertion for this data entry
                    new_story_line = " ".join(new_story_list)
                    new_story_lines.append(new_story_line)
                    new_answer_line = answerTokensToAnswerLine(new_answer_list)
                    new_answer_lines.append(new_answer_line)
                else:
                    story_token_list = new_story_list
                    answer_list = new_answer_list
        else: # if this line is invalid, just skip it
            pass
            # print i
            # print question_lines[i]
            # print answer_lines[i]
        #except:
        #    pass
    # print story_lines[0]
    # print story_lines[1]
    # print story_lines[2]
    write_to_file(adv_train_story_file,new_story_lines)
    write_to_file(adv_train_answer_file,new_answer_lines)
    write_to_file(adv_train_question_file,new_question_lines)
    print "adversarial data generated at "+adv_train_story_filepath


# generate_adversarial_data(original_data_folder_path, train_story_filename, train_question_filename, new_data_folder_path,train_answer_filename)
# generate_adversarial_data(original_data_folder_path, valid_story_filename, valid_question_filename, new_data_folder_path,valid_answer_filename)

def generate_adv_train_and_valid(original_data_folder_path,new_folder_path,insertion_mode,num_insertion):
    # this function will generate both train and valid adv data, then
    generate_adversarial_data(original_data_folder_path, train_story_filename, train_question_filename,
                              new_folder_path,train_answer_filename,insertion_mode, num_insertion)
    generate_adversarial_data(original_data_folder_path, valid_story_filename, valid_question_filename,
                              new_folder_path, valid_answer_filename, insertion_mode, num_insertion)


# for each setting we generate a folder containing the corresponding adv data


# settings = [["../advdata/advdata_front_only","FRONT",1],
#             ["../advdata/advdata_end_only", "END",   1],
#             ["../advdata/advdata_any_1", "ANYPLACE", 1],
#             ["../advdata/advdata_any_2", "ANYPLACE", 2],
#             ["../advdata/advdata_any_3", "ANYPLACE", 3],
#             ["../advdata/advdata_any_4", "ANYPLACE", 4]]
#
# for setting in settings:
#     generate_adv_train_and_valid(original_data_folder_path,setting[0],setting[1],setting[2])
#


# this is used to generate one of the test sets
# we need to use a different function (generate_adversarial_test_data) to deal with multiple answers
generate_adversarial_test_data(original_data_folder_path, test_story_filename, test_question_filename,
                              "../new_test_data",test_answer_filename,"ANYPLACE", 1)



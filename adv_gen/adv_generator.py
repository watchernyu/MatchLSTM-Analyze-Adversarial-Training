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

new_data_folder_path = "../squad_adv_test"

N_ADV_WORDS = 10 # this is the number of random words that we put at the end of the sentence (or elsewhere)

def get_1000_common():
    # this function will return a list,
    # the list contains the 1000 words
    common_1000_file = open("common_1000.txt","r")
    common_words = common_1000_file.read().split("\t")
    print common_words[:10]
    return common_words

COMMON_WORDS = get_1000_common()

def lineToTokens(line):
    # use this function to convert question to tokens, so it's easier to use
    # when we try to generate new word
    tokens = line.split(" ")
    return tokens[:-1]

def questionsToTokens(questions):
    for i in range(len(questions)):
        questions[i] = lineToTokens(questions[i])
    return questions

# New Rule: Generate sequence of new words that contains at least some question words.
def sample_common(common, num_sample):
    # sample 20 words from common
    sampled_common = np.random.choice(common, num_sample)
    return sampled_common

def generate_sequence(sampled_common, question, num_question_words, num_common_words):
    common_num = random_num = random.randint(num_common_words, N_ADV_WORDS - num_question_words)
    common = list(np.random.choice(sampled_common, common_num))
    q = list(np.random.choice(question, N_ADV_WORDS - common_num))
    sequence = common + q 
    random.shuffle(sequence)
    return sequence + ['.']

def generate_new_word(base, num_question_words):
    # generate a new word randomly, this word is picked from either the 1000 common words
    # or from the words in the question (the question is given to the function in the form of tokens)
    
    random_num = random.randint(0,len(base)-1) #Return a random integer x such that a <= x <= b
    word = base[random_num]

    if random_num < n_common:
        word = common[random_num]
    else:
        word = question[random_num%n_common]

    # print total_n
    # print common[:5]
    # print question
    # print random_num
    # print random_num%n_common
    # print word
    return word, num_question_words

def write_to_file(file,listOfLines):
    # this function will write a list of sentences to a txt file
    for line in listOfLines:
        file.write(line+"\n")

def generate_adversarial_data(original_data_folder_path,story_filename,question_filename,new_data_folder_path, answer_filename):
    # this function should be run for training data and validation data

    story_path = os.path.join(original_data_folder_path, story_filename)
    question_path = os.path.join(original_data_folder_path, question_filename)
    answer_path = os.path.join(original_data_folder_path, answer_filename)

    train_story_file = open(story_path,"r")
    story_lines = train_story_file.read().strip().split("\n")

    train_question_file = open(question_path,"r")
    question_lines = questionsToTokens(train_question_file.read().strip().split("\n"))

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

    n = len(story_lines)
    new_story_lines = []
    new_answer_lines = []
    for i in range(n): # for each line
        answer_list = answer_lines[i].strip().split(':')
        if len(question_lines[i]) > 1 and len(answer_list) == 2:
        #try:
            sampled_common = sample_common(COMMON_WORDS, 20)
            sequence = generate_sequence(sampled_common, question_lines[i], 4, 3)
            #print (sequence)
        #for j in range(N_ADV_WORDS): # for each adversarial word
        #    word = generate_new_word(COMMON_WORDS,question_lines[i])
        #    added_line+=word+" "
            story_token_list = story_lines[i].strip().split()
            insert_indices = [i for i, x in enumerate(story_token_list) if x == "."]
            insert_indices = [0] + insert_indices
            insert = np.random.choice(insert_indices,1)[0]

            #print ('insert',insert)
            # compare insert position with answer range to see if we need to change answer range
            answer_start = int(answer_list[0])
            answer_end = int(answer_list[1])
            #print ('answer_start',answer_start)

            # split the story into two parts
            if insert != 0:
                first_half_story = story_token_list[:insert+1]
                second_half_story = story_token_list[insert+1:]
            else:
                first_half_story = []
                second_half_story = story_token_list

            if insert <= answer_start:
                #print ('i',i)
                answer_start += len(sequence)
                answer_end += len(sequence)

                #print ('new_answer_start', answer_start)
            new_story_list = first_half_story + sequence + second_half_story
            new_story_line = " ".join(new_story_list)
            new_story_lines.append(new_story_line)

            new_answer_list = [str(answer_start), str(answer_end)]
            new_answer_line = ":".join(new_answer_list)
            new_answer_lines.append(new_answer_line)
        else:
            print i 
            print question_lines[i]
            print answer_lines[i]
        #except:
        #    pass
    # print story_lines[0]
    # print story_lines[1]
    # print story_lines[2]
    write_to_file(adv_train_story_file,new_story_lines)
    write_to_file(adv_train_answer_file,new_answer_lines)
    print "adversarial data generated at "+adv_train_story_filepath

generate_adversarial_data(original_data_folder_path, train_story_filename, train_question_filename, new_data_folder_path,train_answer_filename)
generate_adversarial_data(original_data_folder_path, valid_story_filename, valid_question_filename, new_data_folder_path,valid_answer_filename)

#
# for line in lines:
#


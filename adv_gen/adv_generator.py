# this program uses the common_1000 txt and the squad tokenized dataset to generate
# some adv training data
import os
import random

original_data_folder_path = "../squad_tokenized_original"
train_question_filename = "train-v1.1-question.txt"
train_story_filename = "train-v1.1-story.txt"
new_data_folder_path = "../squad_adv_0"

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

def generate_new_word(common, question):
    # generate a new word randomly, this word is picked from either the 1000 common words
    # or from the words in the question (the question is given to the function in the form of tokens)
    n_common = len(common)
    total_n = n_common + len(question)
    random_num = random.randint(0,total_n-1) #Return a random integer x such that a <= x <= b
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
    return word

def write_to_file(file,listOfLines):
    # this function will write a list of sentences to a txt file
    for line in listOfLines:
        file.write(line+"\n")


def generate_adversarial_data(original_data_folder_path,story_filename,question_filename,new_data_folder_path):
    # this function should be run for training data and validation data

    story_path = os.path.join(original_data_folder_path, story_filename)
    question_path = os.path.join(original_data_folder_path, question_filename)

    train_story_file = open(story_path,"r")
    story_lines = train_story_file.read().split("\n")

    train_question_file = open(question_path,"r")
    question_lines = questionsToTokens(train_question_file.read().split("\n"))

    print len(story_lines)
    print len(question_lines)

    print story_lines[0]
    print question_lines[0]

    adv_train_story_filepath = os.path.join(new_data_folder_path, story_filename)
    adv_train_story_file = open(adv_train_story_filepath,"w")

    n = len(story_lines)
    for i in range(n): # for each line
        added_line = " "
        for j in range(N_ADV_WORDS): # for each adversarial word
            word = generate_new_word(COMMON_WORDS,question_lines[i])
            added_line+=word+" "
        story_lines[i] = story_lines[i] + added_line
    # print story_lines[0]
    # print story_lines[1]
    # print story_lines[2]
    write_to_file(adv_train_story_file,story_lines)
    print "adversarial data generated at "+adv_train_story_filepath

generate_adversarial_data(original_data_folder_path, train_story_filename, train_question_filename, new_data_folder_path)

#
# for line in lines:
#


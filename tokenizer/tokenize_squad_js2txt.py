# use this file to tokenize squad related json data
# required to use python 3.x and allennlp library
# use the command
# python tokenize_squad_js2txt.py <the file path to json file> <prefix name of the 3 output txt files>
# to convert a json file to 3 txt files (story, question, answer)

from allennlp.data.dataset_readers.squad import *
import argparse

parser = argparse.ArgumentParser(description='Tokenize squad-like json data')

parser.add_argument('jsonpath', type=str,
                   help='json file path')

parser.add_argument('txtprefix', type=str,
                   help='txt file prefix')

args = parser.parse_args()

js_file_path = args.jsonpath # specify path here (I should use args)
txt_prefix = args.txtprefix
txt_file_prefix = args.txtprefix # specify prefix, this will be added to beginning of all txt filenames

print("reading json file.")
squad_reader = SquadReader()
dataset = squad_reader.read(js_file_path)

instances = dataset.instances

print("number of data entry: (# of instances)",len(instances))

passages_fpt = open(txt_prefix+"-story.txt","w")
questions_fpt = open(txt_prefix+"-question.txt","w")
answers_fpt = open(txt_prefix+"-answer-range.txt","w")

for i in range(len(instances)):
    ins = instances[i]
    passage_tokens = ins.fields['passage'].tokens
    question_tokens = ins.fields['question'].tokens
    passage_text = ' '.join(passage_tokens).replace('\n','') + '\n'
    question_text = ' '.join(question_tokens) + '\n'

    spanStart = ins.fields['span_start'].sequence_index
    spanEnd = ins.fields['span_end'].sequence_index+1 # need to have this +1 thing because that's how the model works
    answer_span_text = str(spanStart) + ':' + str(spanEnd) + '\n'

    passages_fpt.write(passage_text)
    questions_fpt.write(question_text)
    answers_fpt.write(answer_span_text)

    if i%500 == 0:
        print(i)

print(len(instances),"entries all finished!")

import nltk
# this program is used to generate a txt file that contains
# the 1000 most common words in the Brown Corpus

from nltk.corpus import brown

from nltk import FreqDist
words = brown.words()
print words
print len(words)
fdist = FreqDist(words)

common_words = fdist.most_common(1000)

# testSent = "I have a problem with this thing"
# fdist = FreqDist(testSent)
# print fdist.most_common(2)
#

print common_words[:10]
common_1000_words_file = open("common_1000.txt","w")
n_common_words = 1000
for i in range(n_common_words):
    word = common_words[i][0]+"\t"
    common_1000_words_file.write(word)

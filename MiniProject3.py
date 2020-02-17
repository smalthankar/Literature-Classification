#Reference Credits: http://computational-linguistics-class.org/assignment5.html


import numpy as np

import collections
from collections import *
from random import random

from math import log10 as ln
import re


# Python code to merge dict using a single
# expression
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def mergeCounters_en(collectionCounter_en, counter, totalSum_en):
    collectionCounter_en += counter
    totalSum_en += sum(counter.values())


def mergeCounters_fr(collectionCounter_fr, counter, totalSum_fr):
    collectionCounter_fr += counter
    totalSum_fr += sum(counter.values())


def mergeCounters_it(collectionCounter_it, counter, totalSum_it):
    collectionCounter_it += counter
    totalSum_it += sum(counter.values())


def normalize_en(counter, collectionCounter_en, totalSum_en):
    mergeCounters_en(collectionCounter_en, counter, totalSum_en)

    sumOfCounters = float(sum(counter.values()))
    return [(character, countCnt / sumOfCounters) for character, countCnt in counter.items()]


def normalize_fr(counter, collectionCounter_fr, totalSum_fr):
    mergeCounters_fr(collectionCounter_fr, counter, totalSum_fr)

    sumOfCounters = float(sum(counter.values()))
    return [(character, countCnt / sumOfCounters) for character, countCnt in counter.items()]


def normalize_it(counter, collectionCounter_it, totalSum_it):
    mergeCounters_it(collectionCounter_it, counter, totalSum_it)

    sumOfCounters = float(sum(counter.values()))
    return [(character, countCnt / sumOfCounters) for character, countCnt in counter.items()]


def train_char_lm_fr(fname, count, collectionCounter_fr, totalSum_fr, typeOfData, order=1):
    unCleanData = open(fname).read()
    # spaceSettledData = re.sub('[ ]+', '$', unCleanData)
    allCaseData = re.sub('[^A-Za-z]+', '', unCleanData)

    data = allCaseData.lower()
    lm = defaultdict(Counter)

    pad = "~" * order
    data = pad + data
    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    outlm = {hist: normalize_fr(chars, collectionCounter_fr, totalSum_fr) for hist, chars in lm.items()}

    c_new_fr = Counter()
    for k, v in collectionCounter_fr.items():
        if k == ' ':
            new_k = re.sub(' ', '$', k)
            c_new_fr.update({new_k: v})

        elif k != '[^A-Za-z]':
            new_k = re.sub('[^A-Za-z]+', '', k)
            c_new_fr.update({new_k.lower(): v})

    if '' in c_new_fr:
        del c_new_fr['']

    if typeOfData == "collection":
        return c_new_fr
    elif typeOfData == "lm":
        return outlm


def train_char_lm_en(fname, count, collectionCounter_en, totalSum_en, typeOfData, order=1):
    unCleanData = open(fname).read()
    # spaceSettledData = re.sub('[ ]+', '$', unCleanData)
    allCaseData = re.sub('[^A-Za-z]+', '', unCleanData)

    data = allCaseData.lower()
    lm = defaultdict(Counter)

    pad = "~" * order
    data = pad + data
    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    outlm = {hist: normalize_en(chars, collectionCounter_en, totalSum_en) for hist, chars in lm.items()}

    c_new_en = Counter()
    for k, v in collectionCounter_en.items():
        if k == ' ':
            new_k = re.sub(' ', '$', k)
            c_new_en.update({new_k: v})

        elif k != '[^A-Za-z]':
            new_k = re.sub('[^A-Za-z]+', '', k)
            c_new_en.update({new_k.lower(): v})

    if '' in c_new_en:
        del c_new_en['']

    if typeOfData == "collection":
        return c_new_en
    elif typeOfData == "lm":
        return outlm


def train_char_lm_it(fname, count, collectionCounter_it, totalSum_it, typeOfData, order=1):
    unCleanData = open(fname, encoding='utf8', errors='ignore').read()
    # spaceSettledData = re.sub('[ ]+', '$', unCleanData)
    allCaseData = re.sub('[^A-Za-z]+', '', unCleanData)

    data = allCaseData.lower()
    lm = defaultdict(Counter)

    pad = "~" * order
    data = pad + data
    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    outlm = {hist: normalize_it(chars, collectionCounter_it, totalSum_it) for hist, chars in lm.items()}

    c_new_it = Counter()
    for k, v in collectionCounter_it.items():
        if k == ' ':
            new_k = re.sub(' ', '$', k)
            c_new_it.update({new_k: v})

        elif k != '[^A-Za-z]':
            new_k = re.sub('[^A-Za-z]+', '', k)
            c_new_it.update({new_k.lower(): v})

    if '' in c_new_it:
        del c_new_it['']

    if typeOfData == "collection":
        return c_new_it
    elif typeOfData == "lm":
        return outlm


totalSum_en = 0
collectionCounter_en = collections.Counter()

typeOfData1 = "collection"
typeOfData2 = "lm"

final_en1 = train_char_lm_en("TrainingCorpusENandFR/en-moby-dick.txt", 'en', collectionCounter_en, totalSum_en,
                             typeOfData1, order=1)
final_en2 = train_char_lm_en("TrainingCorpusENandFR/en-moby-dick.txt", 'en', collectionCounter_en, totalSum_en,
                             typeOfData1, order=1)
lm_en1 = train_char_lm_en("TrainingCorpusENandFR/en-moby-dick.txt", 'en', collectionCounter_en, totalSum_en,
                          typeOfData2, order=1)
lm_en2 = train_char_lm_en("TrainingCorpusENandFR/en-the-little-prince.txt", 'en', collectionCounter_en, totalSum_en,
                          typeOfData2, order=1)

lm_en = collections.Counter()
lm_en = Merge(lm_en1, lm_en2)
final_en = final_en1 + final_en2  # Merge(final_en1, final_en2)

totalSum_en = sum(final_en.values())

# print(final_en1)
# print(final_en2)

# print("English : ")
# print("Total characters: " + str(totalSum_en))
# print(final_en)
# print("\n")
# print(lm_en)


totalSum_fr = 0
collectionCounter_fr = collections.Counter()
final_fr1 = train_char_lm_fr("TrainingCorpusENandFR/fr-le-petit-prince.txt", 'fr', collectionCounter_fr, totalSum_fr,
                             typeOfData1, order=1)
final_fr2 = train_char_lm_fr("TrainingCorpusENandFR/fr-vingt-mille-lieues-sous-les-mers.txt", 'fr',
                             collectionCounter_fr, totalSum_fr, typeOfData1, order=1)
lm_fr1 = train_char_lm_fr("TrainingCorpusENandFR/fr-le-petit-prince.txt", 'fr', collectionCounter_fr, totalSum_fr,
                          typeOfData2, order=1)
lm_fr2 = train_char_lm_fr("TrainingCorpusENandFR/fr-vingt-mille-lieues-sous-les-mers.txt", 'fr', collectionCounter_fr,
                          totalSum_fr, typeOfData2, order=1)

lm_fr = Merge(lm_fr1, lm_fr2)
final_fr = final_fr1 + final_fr2  # Merge(final_fr1, final_fr2)

totalSum_fr = sum(final_fr.values())

# print(final_fr1)
# print(final_fr2)

# print("French : ")
# print("Total characters: " + str(totalSum_fr))
# print(final_fr)
# print("\n")
# print(lm_fr)


# ita-sm_web_2016_10K-sentences


totalSum_it = 0
collectionCounter_it = collections.Counter()
final_it1 = train_char_lm_it("TrainingCorpusENandFR/it-text-1.txt", 'it', collectionCounter_it, totalSum_it,
                             typeOfData1, order=1)
final_it2 = train_char_lm_it("TrainingCorpusENandFR/it-text-2.txt", 'it', collectionCounter_it, totalSum_it,
                             typeOfData1, order=1)
lm_it1 = train_char_lm_it("TrainingCorpusENandFR/it-text-1.txt", 'it', collectionCounter_it, totalSum_it, typeOfData2,
                          order=1)
lm_it2 = train_char_lm_it("TrainingCorpusENandFR/it-text-2.txt", 'it', collectionCounter_it, totalSum_it, typeOfData2,
                          order=1)

lm_it = Merge(lm_it1, lm_it2)
final_it = final_it1 + final_it2  # Merge(final_fr1, final_fr2)

# totalSum_en = sum(final_en.values())

totalSum_it = sum(final_it.values())


# print(final_fr1)
# print(final_fr2)

# print("Italian : ")
# print("Total characters: " + str(totalSum_it))
# print(final_it)
# print(lm_fr)


def checkProbabilityOfCharacter_en(totalSum_en, final_en, lm_en, ch, delta, n_gram):
    if n_gram == 1:
        numberOfOccurance = final_en[ch]
        prob = (numberOfOccurance + delta) / (totalSum_en + delta * 26)
        # print(prob)
        return prob
    elif n_gram == 2:
        # lm_en[ch[0]]
        count = 0
        for nextElement in lm_en[ch[0]]:

            if nextElement[0] == ch[1]:
                # print(nextElement)
                # print(nextElement[1])
                return nextElement[1]
            elif count == len(lm_en[ch[0]]) - 1:
                return (delta) / (final_en[ch[0]] + delta * 26)
            count += 1
    elif n_gram == 3:
        first_part_ch = ch[:2]
        second_part_ch = ch[2:3]
        count = 0
        for nextElement in lm_en[first_part_ch]:
            if nextElement[0] == second_part_ch:
                return nextElement[1]
            elif count == len(lm_en[first_part_ch]) - 1:
                return (delta) / (lm_en[first_part_ch] + delta * 26)
            count += 1


def checkProbabilityOfCharacter_fr(totalSum_fr, final_fr, lm_fr, ch, delta, n_gram):
    if n_gram == 1:
        numberOfOccurance = final_fr[ch]
        prob = (numberOfOccurance + delta) / (totalSum_fr + delta * 26)
        return prob
    elif n_gram == 2:
        count = 0
        for nextElement in lm_fr[ch[0]]:
            if nextElement[0] == ch[1]:
                return nextElement[1]
            elif count == len(lm_fr[ch[0]]) - 1:
                return (delta) / (final_fr[ch[0]] + delta * 26)
            count += 1
    elif n_gram == 3:
        first_part_ch = ch[:2]
        second_part_ch = ch[2:3]
        count = 0
        for nextElement in lm_fr[first_part_ch]:
            if nextElement[0] == second_part_ch:
                return nextElement[1]
            elif count == len(lm_fr[first_part_ch]) - 1:
                return (delta) / (lm_fr[first_part_ch] + delta * 26)
            count += 1


def checkProbabilityOfCharacter_it(totalSum_it, final_it, lm_it, ch, delta, n_gram):
    if n_gram == 1:
        numberOfOccurance = final_it[ch]
        prob = (numberOfOccurance + delta) / (totalSum_it + delta * 26)
        return prob
    elif n_gram == 2:
        count = 0
        for nextElement in lm_it[ch[0]]:
            if nextElement[0] == ch[1]:
                return nextElement[1]
            elif count == len(lm_it[ch[0]]) - 1:
                return ((delta) / (final_it[ch[0]] + delta * 26))
            count += 1
    elif n_gram == 3:
        first_part_ch = ch[:2]
        second_part_ch = ch[2:3]
        count = 0
        for nextElement in lm_it[first_part_ch]:
            if nextElement[0] == second_part_ch:
                return nextElement[1]
            elif count == len(lm_it[first_part_ch]) - 1:
                return ((delta) / (lm_it[first_part_ch] + delta * 26))
            count += 1


def getNgrams(b, n):
    charList = [b[i:i + n] for i in range(len(b) - n + 1)]

    charList = [x.lower() for x in charList]
    return charList


def getSentence(sentence, n_gram, countOfSentence):
    originalSentence = sentence
    sentence = re.sub('[^A-Za-z]+', '', sentence)

    countOfSentence = str(countOfSentence)

    delta = 0.5

    n_gram1 = getNgrams(sentence, 1)

    all_log_prob_fr = []
    all_log_prob_en = []
    all_log_prob_it = []

    with open('Output_Files/out' + countOfSentence + '.txt', 'w') as file:
        file.write('%s\n\n' '%s\n\n' % (originalSentence, "UNIGRAM MODEL:"))

    prob_char_log_fr = 0
    prob_char_log_en = 0
    prob_char_log_it = 0
    for ch in n_gram1:
        prob_char_fr = checkProbabilityOfCharacter_fr(totalSum_fr, final_fr, lm_fr, ch, delta, n_gram)

        prob_char_log_fr = prob_char_log_fr + ln(prob_char_fr)
        all_log_prob_fr.append(prob_char_log_fr)

        prob_char_en = checkProbabilityOfCharacter_en(totalSum_en, final_en, lm_en, ch, delta, n_gram)

        prob_char_log_en = prob_char_log_en + ln(prob_char_en)
        all_log_prob_en.append(prob_char_log_en)

        prob_char_it = checkProbabilityOfCharacter_it(totalSum_it, final_it, lm_it, ch, delta, n_gram)

        prob_char_log_it = prob_char_log_it + ln(prob_char_it)
        all_log_prob_it.append(prob_char_log_it)

        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n' % ("UNIGRAM : " + ch))
            file.write('%s\n' % ("FRENCH: P(" + ch + ")" + " = " + str(
                prob_char_fr) + " ==> log prob of sentence so far: " + str(
                prob_char_log_fr)))
            file.write('%s\n' % ("English: P(" + ch + ")" + " = " + str(
                prob_char_en) + " ==> log prob of sentence so far: " + str(
                prob_char_log_en)))
            file.write('%s\n\n\n' % (
                        "Other: P(" + ch + ")" + " = " + str(prob_char_it) + " ==> log prob of sentence so far: " + str(
                    prob_char_log_it)))

    total_prob_fr = prob_char_log_fr  # sum(all_log_prob_fr)
    total_prob_en = prob_char_log_en  # sum(all_log_prob_en)
    total_prob_it = prob_char_log_it  # sum(all_log_prob_it)

    maxOfLang = max(total_prob_fr, total_prob_en, total_prob_it)

    if (maxOfLang == total_prob_fr):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n\n\n' % ("According to the unigram model, the sentence is in French"))
            print("According to the unigram model, " + originalSentence + ", goes to French!")
    elif (maxOfLang == total_prob_en):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n\n\n' % ("According to the unigram model, the sentence is in English"))
            print("According to the unigram model, " + originalSentence + ", goes to English!")
    elif (maxOfLang == total_prob_it):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n\n\n' % ("According to the unigram model, the sentence is in Other (Italian)"))
            print("According to the unigram model, " + originalSentence + ", goes to Italian!")

    n_gram2 = getNgrams(sentence, 2)

    all_log_prob_fr_bi = []
    all_log_prob_en_bi = []
    all_log_prob_it_bi = []

    with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
        file.write('%s\n\n' '%s\n\n' % ("----------------------------------------", "BIGRAM MODEL:"))

    prob_char_log_fr_bi = 0
    prob_char_log_en_bi = 0
    prob_char_log_it_bi = 0
    for ch in n_gram2:
        prob_char_fr_bi = checkProbabilityOfCharacter_fr(totalSum_fr, final_fr, lm_fr, ch, delta, n_gram + 1)

        prob_char_log_fr_bi = prob_char_log_fr_bi + ln(prob_char_fr_bi)
        all_log_prob_fr_bi.append(prob_char_log_fr_bi)

        prob_char_en_bi = checkProbabilityOfCharacter_en(totalSum_en, final_en, lm_en, ch, delta, n_gram + 1)

        prob_char_log_en_bi = prob_char_log_en_bi + ln(prob_char_en_bi)
        all_log_prob_en_bi.append(prob_char_log_en_bi)

        prob_char_it_bi = checkProbabilityOfCharacter_it(totalSum_it, final_it, lm_it, ch, delta, n_gram + 1)

        prob_char_log_it_bi = prob_char_log_it_bi + ln(prob_char_it_bi)
        all_log_prob_it_bi.append(prob_char_log_it_bi)

        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n' % ("BIGRAM : " + ch))
            file.write('%s\n' % ("FRENCH: P(" + ch[1] + "|" + ch[0] + ")" + " = " + str(
                prob_char_fr_bi) + " ==> log prob of sentence so far: " + str(
                prob_char_log_fr_bi)))
            file.write('%s\n' % ("English: P(" + ch[1] + "|" + ch[0] + ")" + " = " + str(
                prob_char_en_bi) + " ==> log prob of sentence so far: " + str(
                prob_char_log_en_bi)))
            file.write('%s\n\n\n' % ("Other: P(" + ch[1] + "|" + ch[0] + ")" + " = " + str(
                prob_char_it_bi) + " ==> log prob of sentence so far: " + str(
                prob_char_log_it_bi)))

    total_prob_fr_bi = prob_char_log_fr_bi  # sum(all_log_prob_fr_bi)
    total_prob_en_bi = prob_char_log_en_bi  # sum(all_log_prob_en_bi)
    total_prob_it_bi = prob_char_log_it_bi  # sum(all_log_prob_it_bi)

    maxOfLang_bi = 0

    maxOfLang_bi = max(total_prob_fr_bi, total_prob_en_bi, total_prob_it_bi)
    # print(str(maxOfLang_bi) + " " + str(total_prob_fr_bi)  + " " + str(total_prob_en_bi)  + " " + str(total_prob_it_bi))
    if (maxOfLang_bi == total_prob_fr_bi):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n' % ("According to the bigram model, the sentence is in French"))
            print("According to the bigram model, " + originalSentence + ", goes to French!")
    elif (maxOfLang_bi == total_prob_en_bi):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n' % ("According to the bigram model, the sentence is in English"))
            print("According to the bigram model, " + originalSentence + ", goes to English!")
    elif (maxOfLang_bi == total_prob_it_bi):
        with open('Output_Files/out' + countOfSentence + '.txt', 'a') as file:
            file.write('%s\n' % ("According to the bigram model, the sentence is in Other (Italian)"))
            print("According to the bigram model, " + originalSentence + ", goes to Italian!")




def printProbability():
    n_gram = 1

    delta = 0.5

    prob_unigram_en = []
    prob_unigram_fr = []
    prob_unigram_it = []

    prob_bigram_en = []
    prob_bigram_fr = []
    prob_bigram_it = []

    alphabets = []
    for i in range(0, 26):
        alphabets.append(chr(i + 97))
    # print(alphabets)

    concatinated_alphabets_list = []

    for char in alphabets:
        # print("In Loop " + char)
        prob_unigram_character_en = checkProbabilityOfCharacter_en(totalSum_en, final_en, lm_en, char, delta, n_gram)
        prob_unigram_en.append(prob_unigram_character_en)

        prob_unigram_character_fr = checkProbabilityOfCharacter_fr(totalSum_fr, final_fr, lm_fr, char, delta, n_gram)
        prob_unigram_fr.append(prob_unigram_character_fr)

        prob_unigram_character_it = checkProbabilityOfCharacter_it(totalSum_it, final_it, lm_it, char, delta, n_gram)
        prob_unigram_it.append(prob_unigram_character_it)

        for second_character in alphabets:
            concatinated_characters = char + second_character

            concatinated_alphabets_list.append(concatinated_characters)

            prob_bigram_character_en = checkProbabilityOfCharacter_en(totalSum_en, final_en, lm_en,
                                                                      concatinated_characters, delta, n_gram + 1)
            prob_bigram_en.append(prob_bigram_character_en)

            prob_bigram_character_fr = checkProbabilityOfCharacter_fr(totalSum_fr, final_fr, lm_fr,
                                                                      concatinated_characters, delta, n_gram + 1)
            prob_bigram_fr.append(prob_bigram_character_fr)

            prob_bigram_character_it = checkProbabilityOfCharacter_it(totalSum_it, final_it, lm_it,
                                                                      concatinated_characters, delta, n_gram + 1)
            prob_bigram_it.append(prob_bigram_character_it)

    with open('Language_Models_Output_Files/unigramEN.txt', 'w'):
        pass
    with open('Language_Models_Output_Files/unigramFR.txt', 'w') as file:
        pass
    with open('Language_Models_Output_Files/unigramOT.txt', 'w') as file:
        pass
    for char in range(0, 26):
        character_unigram_en = prob_unigram_en[char]
        # print(alphabets[char] + " " + str(character_unigram_en))
        with open('Language_Models_Output_Files/unigramEN.txt', 'a') as file:
            file.write('%s\n' % ("P(" + alphabets[char] + ")" + " = " + str(character_unigram_en)))

        character_unigram_fr = prob_unigram_fr[char]
        with open('Language_Models_Output_Files/unigramFR.txt', 'a') as file:
            file.write('%s\n' % ("P(" + alphabets[char] + ")" + " = " + str(character_unigram_fr)))

        character_unigram_it = prob_unigram_it[char]
        with open('Language_Models_Output_Files/unigramOT.txt', 'a') as file:
            file.write('%s\n' % ("P(" + alphabets[char] + ")" + " = " + str(character_unigram_it)))

    with open('Language_Models_Output_Files/bigramEN.txt', 'w') as file:
        pass
    with open('Language_Models_Output_Files/bigramFR.txt', 'w') as file:
        pass
    with open('Language_Models_Output_Files/bigramOT.txt', 'w') as file:
        pass
    for j in range(0, len(concatinated_alphabets_list)):
        character_con_from_list = concatinated_alphabets_list[j]
        character_1 = character_con_from_list[0]
        character_2 = character_con_from_list[1]

        character_bigram_en = prob_bigram_en[j]
        with open('Language_Models_Output_Files/bigramEN.txt', 'a') as file:
            file.write('%s\n' % ("P(" + character_1 + "|" + character_2 + ")" + " = " + str(character_bigram_en)))

        character_bigram_fr = prob_bigram_fr[j]
        with open('Language_Models_Output_Files/bigramFR.txt', 'a') as file:
            file.write('%s\n' % ("P(" + character_1 + "|" + character_2 + ")" + " = " + str(character_bigram_fr)))

        character_bigram_it = prob_bigram_it[j]
        with open('Language_Models_Output_Files/bigramOT.txt', 'a') as file:
            file.write('%s\n' % ("P(" + character_1 + "|" + character_2 + ")" + " = " + str(character_bigram_it)))


printProbability()


def testSentences(fname):
    n_gram = 1
    
    # fname = "Ten_Sentences_given.txt"
    with open(fname) as f:
        content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    countOfSentence = 0
    for sentence in content:
        print(sentence)
        countOfSentence += 1
        getSentence(sentence, 1, countOfSentence)



testSentences("Ten_Sentences_given.txt")


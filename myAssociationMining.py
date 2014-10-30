# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 13:46:27 2014

@author: Marcel Caraciolo
For more information on this file see:
http://aimotion.blogspot.ca/2013/01/machine-learning-and-data-mining.html

"""

frequent_words = [
    "for",
    "a",
    "of",
    "by",
    "in",
    "to",
    "the",
    "and",
    "was",
    "with",
    "these",
    "were",
    "be",
    "as",
    "is",
    "no",
    "it",
    "at",
    "had",
    "be",
    "can",
    "that",
    "an",
    "less",
    "than",
    "patients",
    "patient",
    "been",
    "have",
    "not",
    "but",
    "than",
    "on",
    "kg",
    "this",
    "we",
    "induced",
    "ng",
    "ml",
    "which",
    "are",
    "from",
    "induced",
    "after",
    "all",
    "",
    "not",
    "but",
    "than",
    "on",
    "kg",
    "this",
    "or"
]
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_dataset():
    import arff
    import string
    "Load the sample dataset."
    rows = list(arff.load('medical.arff'))
    thedata = []
    for row in rows:
        thedata_row = []
        text_line = row.theext

        forbidden_chars =  ".!@#$%^&*()_+-=:;[]{\|}<>,.?/~``\t"
        table = string.maketrans(forbidden_chars, ' ' * len(forbidden_chars))

        #Remove forbidden characters and then move to lower case
        text_line = text_line.translate(table).lower()
        text_tokens = text_line.split(' ')

        #Remove any empty string words from the list
        text_tokens = filter(lambda t: t != '', text_tokens)

        for word in text_tokens:
            if word not in frequent_words and not is_number(word):
                thedata_row.append(word)
        thedata.append(thedata_row)

    return thedata


def createC1(dataset):
    "Create a list of candidate item sets of size one."
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    #frozenset because it will be a ket of a dictionary.
    return map(frozenset, c1)


def scanD(dataset, candidates, min_support):
    "Returns all candidates that meets a minimum support level"
    sscnt = {}
    for tid in dataset:
        for can in candidates:
            if can.issubset(tid):
                sscnt.setdefault(can, 0)
                sscnt[can] += 1

    num_items = float(len(dataset))
    retlist = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_items
        if support >= min_support:
            retlist.insert(0, key)
        support_data[key] = support
    return retlist, support_data


def aprioriGen(freq_sets, k):
    "Generate the joint transactions from candidate sets"
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


def apriori(dataset, minsupport=0.5):
    "Generate a list of candidate item sets"
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, support_data = scanD(D, C1, minsupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minsupport)
        support_data.update(supK)
        L.append(Lk)
        k += 1

    return L, support_data

def generateRules(L, support_data, min_confidence=0.5):
    """Create the association rules
    L: list of frequent item sets
    support_data: support data for those itemsets
    min_confidence: minimum confidence threshold
    """
    rules = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rules_from_conseq(freqSet, H1, support_data, rules, min_confidence)
            else:
                calc_confidence(freqSet, H1, support_data, rules, min_confidence)
    return rules


def calc_confidence(freqSet, H, support_data, rules, min_confidence=0.7):
    "Evaluate the rule generated"
    pruned_H = []
    for conseq in H:
        conf = support_data[freqSet] / support_data[freqSet - conseq]
        if conf >= min_confidence:
            rules.append((freqSet - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H


def rules_from_conseq(freqSet, H, support_data, rules, min_confidence=0.7):
    "Generate a set of candidate rules"
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calc_confidence(freqSet, Hmp1,  support_data, rules, min_confidence)
        if len(Hmp1) > 1:
            rules_from_conseq(freqSet, Hmp1, support_data, rules, min_confidence)

def filter_by_lift(support_data, rules, minlift=25):
    for rule in rules:
        #print rule[2] / support_data[rule[1]]
        if rule[2] / support_data[rule[1]] < minlift:
            rules.remove(rule)
    return rules

def filter_by_interest(support_data, rules, mininterest=50):
    for rule in rules:
        #print  support_data[rule[0] | rule[1]] / (support_data[rule[0]]*support_data[rule[0]])
        if support_data[rule[0] | rule[1]] / (support_data[rule[0]]*support_data[rule[0]]) < mininterest:
            rules.remove(rule)
    return rules

def filter_by_ps(support_data, rules, minps=10):
    for rule in rules:
        #print support_data[rule[0] | rule[1]] - (support_data[rule[0]]*support_data[rule[0]])
        if support_data[rule[0] | rule[1]] - (support_data[rule[0]]*support_data[rule[0]]) < minps:
            rules.remove(rule)
    return rules

def filter_by_phi(support_data, rules, minphi=10):
    for rule in rules:
        p_x = support_data[rule[0]]
        p_y = support_data[rule[1]]
        p_xy = support_data[rule[0] | rule[1]]
        ps = p_xy - (p_x * p_y)
        #print ps / ((p_x * (1-p_x) * p_y * (1-p_y))**0.5)
        if ps / ((p_x * (1-p_x) * p_y * (1-p_y))**0.5) < minphi:
            rules.remove(rule)
    return rules

def print_rules(rules):
    print "Rule........................................." + "." * 39 + "|Confidence"
    for rule in rules:
        x_string = ""
        y_string = ""
        for word in rule[0]:
            x_string += word+" "
        for word in rule[1]:
            y_string += word+" "
        while len(x_string)<40: x_string += "-"
        while len(y_string)<40: y_string += "."

        print "%s--->%s|%f" % (x_string, y_string, rule[2])


the_data = load_dataset()
L,support_data = apriori(the_data,0.002)
rules = generateRules(L,support_data, 0.002)

filtered_rules = filter_by_lift(support_data,rules)
filtered_rules = filter_by_interest(support_data,rules)
filtered_rules = filter_by_ps(support_data,rules)
filtered_rules = filter_by_phi(support_data,rules)

print_rules(filtered_rules)

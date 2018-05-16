import numpy as np
import requests, re

target_url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
text = requests.get(target_url).text

def preprocess(text):
    """
    Given a text, returns a list consisting solely of the word 
    tokens contained therein. Feel free to write helper 
    functions for removing formatting and punctuation, converting 
    word case, creating equivalence classes, etc. 
    
    Parameters
    ----------
    text: string
        A string containing the raw text of your corpus
 
    Returns
    -------
    cleaned_text: list of strings
        a list of the words in your corpus presented in the same 
        order as they appear in text
    """
    cleaned_text = re.sub("[^a-zA-Z0-9.']", " ",  text).split()
    return cleaned_text

    
def calc_ngram_probs(cleaned_text, n=3):
    """
    Calculates the log probability of each unique n-word sequence 
    within a text. 
    
    Parameters
    ----------
    cleaned_text: string
        A list of the words in your corpus presented in the same 
        order as they appear in text
    
    n: int
        The gram-size for your model (i.e., the number of words in 
        your n-gram sequences).
    
    Returns
    -------
    ngram_probs: dict
        A dictionary of key,value pairs where the keys correspond 
        to unique n-gram word sequences and the values correspond 
        to the log probability of the sequence occurring within 
        your corpus
    """
    result = []
    word_dict = {}
    ngram_probs = {}
    for i in range(len(cleaned_text) - n + 1):
        curr_tuple = cleaned_text[i:i+n]
        result.append(curr_tuple)
    for i in result:
        if not (str(i[0]) + " " + str(i[1])) in word_dict:
            word_dict[(str(i[0]) + " " + str(i[1]))] = [i[2]]
        else:
            word_dict[(str(i[0]) + " " + str(i[1]))].append(i[2])
    for key, value in word_dict.items():
        for i in value:
            ngram_probs[key + " " + i] = value.count(i)/len(value)
    return ngram_probs
    

def generate(ngram_probs, seed):
    """
    Completes a sentence using the probabilities using your ngram model.
    
    Parameters
    ----------
    ngram_probs: dict
        A dictionary of key,value pairs where keys correspond 
        to n-gram word sequences and values correspond to the 
        log probability of the sequence occurring within your 
        corpus
    
    seed: list
        A list containing the first n words of a sentence.
    
    Returns
    -------
    sentence: list
        A list of length >= n+1 word tokens constituting a sentence. The 
        n+1st to the last word should be generated using the probabilities
        of your ngram model.
    """
    sentence = seed
    n = len(list(ngram_probs.keys())[0].split())
    abbr = ['mr.', 'mrs.', 'ms.', 'dr.', 'col.', 'sgt.', 'st.', 'esq.']
    for x in range(1000):
        poss, probs = [], []
        if len(sentence) > n + 1:
            for i in sentence:
                if "." in i and i.lower() not in abbr:
                    return sentence
        for key, value in ngram_probs.items():
            word = [""]
            if sentence[-(n-1):] == key.split()[:(n-1)]:
                poss.append(key.split()[n-1])
                probs.append(value)
                if round(sum(probs), 5) == 1:
                    word = np.random.choice(poss, 1, p=probs)
                    sentence.append(word[0])
                    continue
    return
    
    
def log_sentence_prob(ngram_probs, sentence):
    """
    Returns the log probability of a sentence using the probabilites
    of your n-gram model. You may assume that the probability of generating 
    the first n-1 words (i.e., the seed words) is 1.
    
    Parameters
    ----------
    ngram_probs: dict
        A dictionary of key,value pairs where keys correspond 
        to n-gram word sequences and values correspond to the 
        log probability of the sequence occurring within your 
        corpus
    
    sentence: list
        A list of word tokens constituting a single sentence
    
    Returns
    -------
    prob: float
        The log probability of generating the words in sentence
        according to your n-gram model.
    """
    prob = 1
    n = len(list(ngram_probs.keys())[0].split())
    for i in range(1, len(sentence) - n + 1):
        curr = sentence[i:i+n]
        val = ngram_probs[curr[0] + " " + curr[1] + " " + curr[2]]
        prob = prob * val
    return prob

n = 3
seed = ["Today", "I", "went"]

cleaned_text = preprocess(text)
ngram_probs = calc_ngram_probs(cleaned_text, n)
sentence = generate(ngram_probs, seed)
log_prob = log_sentence_prob(ngram_probs, sentence)

print('Sentence: {}'.format(sentence))
print('log P(Sentence): {}'.format(log_prob))

n = 5
seed = ["I", "said", "that"]

ngram_probs = calc_ngram_probs(cleaned_text, n)
sentence = generate(ngram_probs, seed)
log_prob = log_sentence_prob(ngram_probs, sentence)

print('Sentence: {}'.format(sentence))
print('log P(Sentence): {}'.format(log_prob))

n = 7
seed = ["However", "there", "was"]

ngram_probs = calc_ngram_probs(cleaned_text, n)
sentence = generate(ngram_probs, seed)
log_prob = log_sentence_prob(ngram_probs, sentence)

print('Sentence: {}'.format(sentence))
print('log P(Sentence): {}'.format(log_prob))

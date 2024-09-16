import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import defaultdict, Counter
import random

nltk.download('brown')
nltk.download('punkt')

text = brown.words(categories='news')

tokens = [word.lower() for word in text]

n = 4
n_grams = list(ngrams(tokens, n))

model = defaultdict(Counter)
for n_gram in n_grams:
    prefix, next_word = tuple(n_gram[:-1]), n_gram[-1]
    model[prefix][next_word] += 1


def generate_sentence(start_words, num_words):
    result = list(start_words)
    seen_phrases = set()
    for _ in range(num_words):
        prefix = tuple(result[-(n - 1):])
        next_word = random.choices(list(model[prefix].keys()), list(model[prefix].values()))[0]

        phrase = ' '.join(result[-(n - 1):] + [next_word])
        if phrase in seen_phrases:
            continue
        seen_phrases.add(phrase)

        result.append(next_word)

        if next_word in ['.', '!', '?'] and len(result) > 10:
            break

    sentence = ' '.join(result)

    sentence = sentence.replace('..', '.').replace(' .', '.').replace(' ,', ',')
    sentence = sentence.replace('. .', '.').replace(' i.r.s..', ' i.r.s.')

    return sentence


start_words = ("the", "president", "said")
sentence = generate_sentence(start_words, 50)
print(sentence.capitalize())


"""
The president said he will ask congress to increase grants to states to help pay medical bills of the needy aged.
"""

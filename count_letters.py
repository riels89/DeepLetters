
# top 250 words:
{'a': 72, 'b': 19, 'c': 19, 'd': 24, 'e': 131, 'f': 17, 'g': 25, 'h': 65, 'i': 54,
 'j': 2, 'k': 16, 'l': 52, 'm': 42, 'n': 67, 'o': 95, 'p': 16, 'q': 1, 'r': 62,
 's': 51, 't': 91, 'u': 36, 'v': 13, 'w': 36, 'x': 1, 'y': 30, 'z': 0}

# top 500 words:
{'a': 176, 'b': 42, 'c': 75, 'd': 72, 'e': 324, 'f': 51, 'g': 53, 'h': 113, 'i': 135,
 'j': 3, 'k': 25, 'l': 135, 'm': 79, 'n': 145, 'o': 195, 'p': 53, 'q': 2, 'r': 169,
 's': 118, 't': 180, 'u': 73, 'v': 32, 'w': 58, 'x': 5, 'y': 56, 'z': 0}

import pandas as pd

letters = list('abcdefghijklmnopqrstuvwxyz')
letter_counts = {letter: 0 for letter in letters}
words = pd.read_csv('top_500_words.csv')['word']
counter = 0
for word in words:
    if counter < 250:
        for letter in word.lower():
            if letter in letters:
                letter_counts[letter] += 1
    else:
        break
    counter += 1
print(letter_counts)


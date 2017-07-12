#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:18:14 2017

@author: Takanori
"""

"""
current.py
"""
"""""""""""""""""""""""""""""""""
###### 行列の作成について確認  ######
作成してる行列はbag-of-wordsかつ0 or 1
"""""""""""""""""""""""""""""""""

import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

# rstrip()で最後の\nの取り除き
# 全ての本のタイトルを読み込み
titles = [line.rstrip() for line in open('all_book_titles.txt')]

# stopwordsの読み込み
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
        'introduction', 'edition', 'series', 'application',
        'approach', 'card', 'access', 'package', 'plus', 'etext',
        'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
        'third', 'second', 'fourth',
})

print(titles[0:2])

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # 初版, 第二版などの取り除き
    return tokens

word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
for title in titles:
    try:
        # title = title.encode('ascii', 'ignore') # これを入れると何も入らない
        all_titles.append(title)
        tokens = my_tokenizer(title)
        # print('tokens:'+str(tokens)) # token化
        all_tokens.append(tokens)
        # 何番目にtokenが出てきたかを辞書形式で作成
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index # 何番目にtokenが出てきたかを辞書形式で作成
                current_index += 1
                index_word_map.append(token)
    except:
        pass

# now let's create our input matrices
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x


# 行列の作成
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))

i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1
    
print(X.shape)
print(X[:,0])
print(X[:,1])
print(X[:,2])


"""
svd = TruncatedSVD()
Z = svd.fit_transform(X) # 次元削減
plt.scatter(Z[:,0], Z[:,1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
# plt.show()
"""














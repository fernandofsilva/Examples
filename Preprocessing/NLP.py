#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""
# Import the regex module
import re

my_string = """Let's write RegEx!  Won't that be fun?  I sure think so.  
             Can you find 4 sentences?  Or perhaps, all 19 words?"""

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))
del sentence_endings

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))
del capitalized_words

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))
del spaces

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))
del digits, my_string

# Import necessary modules
from nltk.tokenize  import sent_tokenize
from nltk.tokenize  import word_tokenize

# load file
path = "/home/esssfff/Documents/Github/Examples/Datasets/"

with open(path+"grail.txt", "r") as file:
    scene_one = file.read()

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)
del tokenized_sent, unique_tokens

# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
del pattern1, pattern2, sentences, scene_one

# Import the necessary modules
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import regexp_tokenize

tweets = ['This is the best #nlp exercise ive found online! #python', 
    '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"

# Use the pattern on the first tweet in the tweets list
regexp_tokenize(tweets[0], pattern1)

# Write a pattern that matches both mentions and hashtags
pattern2 = r"([#|@]\w+)"

# Use the pattern on the last tweet in the tweets list
regexp_tokenize(tweets[-1], pattern2)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)

del pattern1, pattern2, all_tokens, tweets

# german_text = """Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕"""

# Tokenize and print all words in german_text
#all_words = word_tokenize(german_text)
#print(all_words)

# Tokenize and print only capital words
#capital_words = r"[A-ZÜ]\w+"
#print(regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
#emoji = """['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\
#    U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"""
#print(regexp_tokenize(german_text, emoji))

#del all_words, capital_words, emoji, german_text

# Load libraries
import matplotlib.pyplot as plt

# load file
path = "/home/esssfff/Documents/Github/Examples/Datasets/"

with open(path+"grail.txt", "r") as file:
    holy_grail = file.read()

# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, "\w+") for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()

del holy_grail, line_num_words, lines, pattern, tokenized_lines

# Import Counter
from collections import Counter

# load file
path = "/home/esssfff/Documents/Github/Examples/Datasets/"

with open(path+"wiki_text_debugging.txt", "r") as file:
    article = file.read()

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]
del tokens

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))
del bow_simple

# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

with open(path+"english_stopwords.txt", "r") as file:
    english_stops = file.read()

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]
del lower_tokens

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]
del alpha_only

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
del wordnet_lemmatizer, no_stops

# Create the bag-of-words: bow
bow = Counter(lemmatized)
del lemmatized

# Print the 10 most common tokens
print(bow.most_common(10))
del bow, article

# Load files
import glob

wiki = []

file_list = glob.glob(path + '/Wikipedia_articles/*.txt')
for file in file_list:
    with open(file, "r") as f_input:
        wiki.append(f_input.read())
del file_list, file

articles = []

# Tokenize the article: tokens
for key in wiki:
    tokens = word_tokenize(key)
    lower_tokens = [t.lower() for t in tokens]
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    no_stops = [t for t in alpha_only if t not in english_stops]
    articles.append(no_stops)
del key, wiki, tokens, lower_tokens, alpha_only, no_stops, english_stops

# Import Dictionary
from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))
del computer_id

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]
del articles

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])

# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
del doc

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)
del bow_doc, word_id, word_count
    
from collections import defaultdict
import itertools

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
del word_id, word_count

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 
del total_word_count

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)
del word_id, word_count, sorted_word_count

# Import TfidfModel
from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[corpus[4]]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
del corpus, sorted_tfidf_weights, term_id, tfidf_weights, weight

# Import librarie
import nltk

# Load file
with open(path+"/News articles/uber_apple.txt", "r") as file:
    article = file.read()

# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]
del sentences

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 
del token_sentences

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
del pos_sentences

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
del sent, chunk

# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
del sent, chunk, chunked_sentences
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()
del values, labels, ner_categories

# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
del article

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv(path+"fake_or_real_news.csv")

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train[:5])

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
del score, cm, pred, 

from sklearn.model_selection import GridSearchCV
import numpy as np

# We set random_state=0 for reproducibility 
nb_classifier = MultinomialNB()

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha': np.arange(0, 1, 0.1)}
searcher = GridSearchCV(nb_classifier, parameters, cv=10)
searcher.fit(tfidf_train, y_train)
y_pred = searcher.predict(tfidf_test)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
#print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
#print("Recall:",metrics.recall_score(y_test, y_pred))

# Fiting the model with the best parameters

# Instantiate the classifier: nb_classifier
nb_classifier = MultinomialNB(alpha=0.1)

# Fit to the training data
nb_classifier.fit(tfidf_train, y_train)

# Predict the labels: pred
pred = nb_classifier.predict(tfidf_test)

# Compute accuracy: score
score = metrics.accuracy_score(y_test, pred)

# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])









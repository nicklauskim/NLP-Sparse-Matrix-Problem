# Import libraries
from math import log
# Data manipulation
import numpy as np
import pandas as pd
import string
import re
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Natural langauge processing
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


# Set plot settings for later:
plt.style.use('fivethirtyeight')
sns.set_palette('muted')

# Set font sizes for plots:
plt.rc('font', size = 12)
plt.rc('axes', labelsize = 8)
plt.rc('legend', fontsize = 18)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)


# Import data
df = pd.read_csv('~/Documents/Winter 2022/STATS 202B/Final Project/data/charles.csv', index_col = 0)


# Make all characters lowercase and remove numbers + punctuation
def remove_noise(text):
    text_lc = ''.join([word.lower() for word in text if word not in string.punctuation])
    text = re.sub(r'\d+', '', text_lc)
    return text
df['text'] = df['text'].apply(lambda x: remove_noise(x))

def tokenize(text):
    return re.split('\W+', text)
df['text'] = df['text'].apply(lambda x: tokenize(remove_noise(x)))

stop_words = set(stopwords.words('english'))
stop_words.update(['im', 'let', 'us', ' '])    # TO DO: can add other words we don't want here

def remove_stopwords(text):    
    text_no_stopwords = [word for word in text if word not in stop_words]
    return text_no_stopwords

df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

df['text'] = df['text'].apply(lambda x: x[0:50])
df['text'] = df['text'].apply(lambda x: ' '.join(x))





#Helper functions to be used later:
def preprocess(text):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # List of common stopwords in the english language
    stop_words.update(['im', 'let', 'us', '']) # Add some words we want to eliminate that might not be in the list
    stemmer = nltk.PorterStemmer()
    text_lc = ''.join([word.lower() for word in text if word not in string.punctuation])  # tokenize + make lowercase
    text_rc = re.sub(r'\d+', '', text_lc)  # remove punctuation, other non-letter characters
    tokens = re.split('\W+', text_rc)  # tokenization
    text_cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]  # remove stopwords and stemming
    return text_cleaned

def find_likelihood(word, category):
    """
    Takes as inputs:
    word = which word we want the conditional probability for;
    df = the dataframe of counts we will use (e.g. vect_stats)
    """
    # Follow procedure as before, getting the (subset of the) vectorized df for only 1 category
    category_df = vect_train.loc[np.array(Y_train == category), :]
    n = category_df.shape[0]
    return (category_df[word].sum() + 1) / (n + 1)

def naivebayes(title, categories):
    """
    Takes as inputs a title and a category we want to find the posterior for (both string inputs)
    - Make sure to use proper name of category
    Finds a single probability, denoting the posterior probability, i.e. P(title|category)
    Gives as output a table/dataframe storing the posteriors for every category so we can compare values easily
    """
    # Initialize empty df to fill and return later
    df_nb = pd.DataFrame(np.zeros((1, len(categories))), columns=categories)
    # Preprocess title to get list of words to check
    words = preprocess(title)
    # We want to find the posterior for every category - loop through list of category names
    for cat in categories:
        # Find (product of) likelihoods of the words found in the title (or sum of log probabs?)
        likelihood_product = 0
        for word in words:
            if word in freq_final.columns.values:
                likelihood_product += log(freq_final.loc[cat, word])
        # Multiply by prior (found earlier) to get posterior
        df_nb[cat] = log(priors[cat]) + likelihood_product
    return df_nb



# Rearrange data frame (rename columns, categories)
df = df[['author', 'text']]
df.loc[df['author'] == 2, 'author'] = 'Darwin'
df.loc[df['author'] == 3, 'author'] = 'Dickens'


# Separate df into multiple dfs by author - mainly convenient for plotting
df_darwin = df.loc[np.array(df['author'] == 'Darwin'), :]
df_dickens = df.loc[np.array(df['author'] == 'Dickens'), :]


# Get all words appearing in Darwin+Dickens
all_darwin_words = sorted(preprocess(df_darwin['text']))
all_dickens_words = sorted(preprocess(df_dickens['text']))


# Get the highest ranking words for plotting
#Frequency of Dickens words
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_dickens_words)).items(), key = lambda x: x[1], reverse = True)}
# Get the highest ranking words
dickens_keys = list(words_dict)[:50]
dickens_values = [words_dict[key] for key in list(words_dict)[:50]]

# Likewise for Darwin
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_darwin_words)).items(), key = lambda x: x[1], reverse = True)}
darwin_keys = list(words_dict)[:50]
darwin_values = [words_dict[key] for key in list(words_dict)[:50]]


# EDA
fig, ax = plt.subplots(figsize = (18, 8))

# Plot bar graph of dickens_keys commonly appearing words for given category
sns.barplot(x = dickens_keys, y = dickens_values, palette = 'viridis', ax = ax)
ax.set_xticklabels(dickens_keys, rotation = 90)
#x.set_ylabel("# of Appearances", labelpad = 25)
ax.set_title("Most Common Dickens Words")

fig, ax = plt.subplots(figsize = (18, 8))

# Plot bar graph of dickens_keys commonly appearing words for given category
sns.barplot(x = darwin_keys, y = darwin_values, palette = 'viridis', ax = ax)
ax.set_xticklabels(darwin_keys, rotation = 90)
#x.set_ylabel("# of Appearances", labelpad = 25)
ax.set_title("Most Common Darwin Words")


fig, ax = plt.subplots(figsize = (20, 20))
# Create and generate a word cloud image:
text = " ".join(word for word in all_darwin_words)
wordcloud = WordCloud(max_font_size = 40, max_words = 50, background_color = "white", width = 500, height = 300, collocations = False).generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize = (20, 20))
# Create and generate a word cloud image:
text = " ".join(word for word in all_dickens_words)
wordcloud = WordCloud(max_font_size = 40, max_words = 50, background_color = "white", width = 500, height = 300, collocations = False).generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()


# Split data into training set and test set
X = df[['text']]
Y = df['author']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4)


#Find the distribution of train and test data as well as how many classes in train and test data
PIE1 = sum(Y_train.str.count("Darwin")), sum(Y_train.str.count("Dickens"))
PIE2 = sum(Y_test.str.count("Darwin")), sum(Y_test.str.count("Dickens"))
PIE3 = len(Y_train), len(Y_test)


# Plot pie chart to compare how much data we have in train and test, as well as the distribution between classes
colors = ("orange", "cyan")
labels = 'Darwin', 'Dickens'
colors1 = ("blue", "red")
labels1 = 'Train Data', 'Test Data'
fig = plt.figure()
#ax1 = plt.subplot2grid((1,2),(0,0))
plt.pie(PIE1,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Classes in Train Data:'+str(sum(PIE1)))
fig = plt.figure()
#ax1 = plt.subplot2grid((1,2),(0,1))
plt.pie(PIE2,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Classes in Test Data: '+str(sum(PIE2)))
fig = plt.figure()
plt.pie(PIE3,colors=colors1, labels=labels1, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Train vs Test Data: '+str(sum(PIE3)))



# Implement algorithms from sklearn
print(" *********** Results from sklearn models: *********** ")
for i in range(0,2):
    if i == 0: #i = 0: CountVectorizer
        print(" -------- Results from CountVectorizer: -------- ")
        Vectorizer = CountVectorizer(analyzer = preprocess)
        Vectorizer.fit(X_train['text'])
        text_vect_train_init = Vectorizer.transform(X_train['text'])
        text_vect_test_init = Vectorizer.transform(X_test['text'])
        text_vect_train = Normalizer().fit_transform(text_vect_train_init) #Normalization (mean = 0, st dev = 1)
        text_vect_test = Normalizer().fit_transform(text_vect_test_init)
        text_vect_train_init = pd.DataFrame(text_vect_train_init.toarray(), columns = Vectorizer.get_feature_names())
        text_vect_test_init = pd.DataFrame(text_vect_test_init.toarray(), columns = Vectorizer.get_feature_names())
        pd.DataFrame(text_vect_train_init).to_csv('./data/count_train.csv', index = False)
        pd.DataFrame(text_vect_test_init).to_csv('./data/count_test.csv', index = False)
        
        # n_components: Desired dimensionality of output data. 
        #Must be strictly less than the number of features. 
        #The default value is useful for visualisation. 
        #For LSA, a value of 100 is recommended.
        svd = TruncatedSVD(n_components=100)
        x = np.arange(100)
        
        text_vect_train = svd.fit_transform(text_vect_train) #Singular Value decomposition, i.e. dimension reduction
        text_vect_test = svd.transform(text_vect_test)
        
        print(svd.explained_variance_ratio_.sum())
    
        pd.DataFrame(text_vect_train).to_csv('./data/count_train_svd.csv', index = False)
        pd.DataFrame(text_vect_test).to_csv('./data/count_test_svd.csv', index = False)
        
    else:  #i = 1: TfidfVectorizer
        print(" -------- Results from TF-IDFVectorizer: -------- ")
        Vectorizer = TfidfVectorizer(analyzer = preprocess)
        Vectorizer.fit(X_train['text'])
        text_vect_train_init = Vectorizer.transform(X_train['text'])
        text_vect_test_init = Vectorizer.transform(X_test['text'])
        text_vect_train = Normalizer().fit_transform(text_vect_train_init) #Normalization (mean = 0, st dev = 1)
        text_vect_test = Normalizer().fit_transform(text_vect_test_init)
        text_vect_train_init = pd.DataFrame(text_vect_train_init.toarray(), columns = Vectorizer.get_feature_names())
        text_vect_test_init = pd.DataFrame(text_vect_test_init.toarray(), columns = Vectorizer.get_feature_names())
        pd.DataFrame(text_vect_train_init).to_csv('./data/tfidf_train.csv', index = False)
        pd.DataFrame(text_vect_test_init).to_csv('./data/tfidf_test.csv', index = False)
        
        # n_components: Desired dimensionality of output data. 
        #Must be strictly less than the number of features. 
        #The default value is useful for visualisation. 
        #For LSA, a value of 100 is recommended.
        svd = TruncatedSVD(n_components=100)
        x = np.arange(100)
        
        text_vect_train = svd.fit_transform(text_vect_train) #Singular Value decomposition, i.e. dimension reduction
        text_vect_test = svd.transform(text_vect_test)
    
        print(svd.explained_variance_ratio_.sum())
    
        pd.DataFrame(text_vect_train).to_csv('./data/tfidf_train_svd.csv', index = False)
        pd.DataFrame(text_vect_test).to_csv('./data/tfidf_test_svd.csv', index = False)

    pd.DataFrame(Y_train).to_csv('./data/y_train.csv', index = False)
    pd.DataFrame(Y_test).to_csv('./data/y_test.csv', index = False)
        
    fig = plt.figure()
    plt.plot(x, svd.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('SV')
    plt.ylabel('Variance Explained')
    plt.show()

    # Create dataframes (both train and test) of vectorized text
    vect_train = pd.DataFrame(text_vect_train)
    vect_test = pd.DataFrame(text_vect_test)
    
    # Logistic Regression
    clf = LogisticRegression(random_state=0)
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    # Use score function to get the accuracy
    print("Logistic Regression Accuracy Score: ", clf.score(vect_test, Y_test))
    
    # NB classifier
    clf = naive_bayes.GaussianNB()
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score: ", accuracy_score(predictions, Y_test))

    # SVM:
    clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test) # Predict the labels on the test dataset
    print("SVM Accuracy Score: ", accuracy_score(predictions, Y_test)) 

    #Random Forest:
    clf = RandomForestClassifier()
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    print("Random Forest Accuracy Score: ", accuracy_score(predictions, Y_test))

    #k-nearest neighbors:
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    print("kNN Accuracy Score: ", accuracy_score(predictions, Y_test))
    
    
    



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import tokenize,ngrams
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
#nltk.download('vader_lexicon')

review = pd.read_csv('review_eda.csv')
review.columns = ['review_url', 'rating', 'review_title', 'user_name', 'user_url',
                  'review_date', 'helpful', 'spoiler', 'review_text']
review = review.drop_duplicates()
review = review.reset_index(drop=True)

# clean review text
p = '[\d,]* out of [\d,]* found this helpful.*\nPermalink'
review.review_text = review.review_text.apply(lambda x: re.sub(p, '', x))
review.review_text.apply(lambda x: 'found this helpful. Was this review helpful? Sign in to vote.' in x).sum()
#Let us also look at the days of the week the review is posted.
review['review_day'] = pd.to_datetime(review['review_date']).dt.day_name()
review['review_day_no'] = pd.to_datetime(review['review_date']).dt.dayofweek
#5. Create Feature: review lemma
w_tokenizer = tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower()) if w not in stopwords.words('english')]
review['review_lemmas'] = review['review_text'].apply(lambda x : lemmatize_text(x))
#created string from review lemmas
review['review_lemmas_str']=review['review_lemmas'].apply(lambda x: ' '.join(x))

analyzer = SentimentIntensityAnalyzer()
review['polarity'] = review['review_lemmas_str'].apply(lambda x: analyzer.polarity_scores(x))
review = pd.concat([review,review['polarity'].apply(pd.Series)], axis=1)
review['sentiment'] = review['compound'].apply (lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')

def plots():
    fig, ax = plt.subplots(2,2,figsize=(50,30))
    plt.xticks(rotation = 90)
    sns.countplot(x= review.rating,order = review.rating.value_counts().index,ax=ax[0,0])
    sns.countplot(x=review.review_day,order=review.review_day.value_counts().index,ax=ax[0,1]) 
    sns.countplot(x=review.review_day_no,order=review.review_day_no.value_counts().index,ax=ax[1,0] )
    sns.countplot(x=review.sentiment, palette=['#b2d8d8',"#008080", '#db3d13'],ax= ax[1,1])
    st.pyplot(fig)
    fig= plt.figure(figsize=(50, 40))        
    #curr_lemmatized_tokens = list(review['review_lemmas'])
    vectorizer = CountVectorizer(ngram_range=(2,2))
    bag_of_words = vectorizer.fit_transform(review['review_lemmas'].apply(lambda x : ' '.join(x)))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 100
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_dict)
  
    plt.imshow(wordCloud)
    plt.title('Word Cloud')
    plt.axis("off")
    st.pyplot(fig)



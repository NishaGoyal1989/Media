import utils 
import pandas as pd
import re
import csv
from selenium import webdriver
import string
import pickle
from selenium.webdriver.common.by import By
from gensim import models
import numpy as np
import pandas as pd
import re
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
from wordcloud import WordCloud
import csv
#import nltk
import string
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn import metrics
import warnings
from gensim import models

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import movie_recom
import streamlit as st
import review_analysis
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

def preprocessing(url):
    data = utils.get_movie_info(url)
    with open('url_info.txt','w') as f:
        f.write('Title\tRating\tGenre\tLength\tRelease_Date\tDirector\tWriter\tStar\tBudget\tReview_URL\n')
        try: 
            f.write('\t'.join(str(d) for d in data) + '\n')
        except:
             print('Failed scraping: {}'.format(url))
    movie= pd.read_csv('url_info.txt',sep='\t')
    movie['Movie_ID'] = movie.Review_URL.apply(lambda x: x.split('/')[4])
    # Parse budget into budget_currency and budget_value
    p = re.compile('(.+?)\d')
    #budget_currency = movie.Budget.apply(lambda x: p.match(x).group(1).strip())
    budget_value = movie.Budget.apply(lambda x: int(''.join([d for d in x if d.isdigit()])))
    movie['Budget_Final']=budget_value
    #movie['Box_Office_Value'] = movie.Box_Office_Gross.apply(lambda x: int(''.join([d for d in x if d.isdigit()])))
    movie['Length_Min'] = movie.Length.apply(utils.parse_time)
    # get movie_id from url
    movie['Movie_ID'] = movie.Review_URL.apply(lambda x: x.split('/')[4])
    # convert release date into date type
    movie.Release_Date = pd.to_datetime(movie.Release_Date.apply(lambda x: re.sub(' \(.*\)', '', x)), errors='coerce')

    
    review_url= movie.Review_URL[0]
    driver = webdriver.Chrome()
    with open('review_info.csv', 'w') as f:
        writer = csv.writer(f)
        try:
            driver.get(review_url)
            # try:
            #     utils.load_page(driver)
            # except:
            #     pass 
            reviews = driver.find_elements(By.CSS_SELECTOR,'div.lister-item')
            for review in reviews:
                data = [review_url] + utils.get_review_info(review)
                writer.writerow(data)
        except:
                print('Failed: ', review_url)
    
    review = pd.read_csv('review_info.csv')
    review.columns = ['review_url', 'rating', 'review_title', 'user_name', 'user_url',
                  'review_date', 'helpful', 'spoiler', 'review_text']
    review = review.drop_duplicates()
    review = review.reset_index(drop=True)
    review.dropna(subset=['review_text'],inplace=True)
    review.review_date = pd.to_datetime(review.review_date)
    # merger table
    review = review.merge(movie, left_on='review_url', right_on='Review_URL')
    # we found that there are several reviews contain unexpected text
    review.review_text.apply(lambda x: 'found this helpful. Was this review helpful? Sign in to vote.' in x).sum()
    # clean review text
    p = '[\d,]* out of [\d,]* found this helpful.*\nPermalink'
    review.review_text = review.review_text.apply(lambda x: re.sub(p, '', x))
    review.review_text.apply(lambda x: 'found this helpful. Was this review helpful? Sign in to vote.' in x).sum()
    # get the total number of review for each movie
    review_count = len(review)
    movie['review_count'] = review_count
    review_length = review.review_text.apply(lambda x: len(x.split()))
    movie['avg_review_length'] = review_length.mean()
    movie.review_count.fillna(0, inplace=True)
    movie.avg_review_length.fillna(0, inplace=True)
    review.rating = pd.to_numeric(review.rating.apply(lambda x: x.split('/')[0] if type(x) == str else None))
    movie['avg_rating'] = str(review['rating'].mean())
    crew = []
    for i, r in movie.iterrows():
        movie_id = r.Movie_ID
        for director in r.Director.split('/'):
        
            crew.append([director, 'director',movie_id])
        for writer in str(r.Writer).split('/'):
        
            crew.append([writer, 'writer',movie_id])
        for star in r.Star.split('/'):
        
            crew.append([star, 'star',movie_id])
    crew = pd.DataFrame(crew, columns=['Person', 'Role','Movie_ID'])
    person = pd.read_csv('person_preprocessec.csv')
    person_info = crew.merge(person ,left_on=['Person','Role'],right_on=['Person','Role'])
    
    person_info.drop(['Unnamed: 0','Movie_ID_y'],axis=1,inplace=True)
    person_info.rename(columns = {'Movie_ID_x':'Movie_ID','Movie_ID':'Movie_Id'}, inplace = True)
    person_info = person_info[['Person','Role','person_value','Movie_ID']].groupby('Role').max()
    person_info['Role']= person_info.index
    person_info=person_info.reset_index(drop=True)
    
    # def role_value(x):
    #     value=[]
        
    #     x= x.split('/')
    #     for item in x:
    #         if len(person_info.person_value[person_info['Person']==item])!= 0 :
    #             value.append(person_info.person_value[person_info['Person']==item][0])
            
    #     return max(value)
    for role in ['director','writer','star']:
        movie[role + '_value'] =  person_info.person_value[(person_info.Role == role)].max() 
        
    # sentences = [ [token.strip(string.punctuation).strip() \
    #             for token in nltk.word_tokenize(doc.lower()) \
    #              if token not in string.punctuation and \
    #              len(token.strip(string.punctuation).strip())>=2]\
    #             for doc in review.review_text]

    wv_model = models.KeyedVectors.load_word2vec_format('numberbatch-en-17.06.txt', binary=False)
    # expand aspect clue word list using word2vec model
    aspects = {
    'acting': {'chemistry', 'performance', 'charm', 'comedian'},
    'direction': {'director', 'filmmaker', 'vision'},
    'screenplay': {'sequence', 'script', 'lines', 'editing', 'screenwriting'}, 
    'sound': {'score', 'music', 'vocals', 'audio'},
    'story': {'mystery', 'spoof', 'thriller', 'twist', 'shock'},
    'visual': {'effects', '3d', 'scenery', 'photography', 'camera', 'cinematography'},
    'film': {'flick', 'remake', 'sequel', 'classic', 'entertainment'}
}
    for k in aspects:
        similar_words = wv_model.most_similar(positive=list(aspects[k]), topn=10)
        aspects[k].update({w[0] for w in similar_words})
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    
    director = person_info[person_info.Role == 'director']
    star = person_info[person_info.Role == 'star']
    
    director_count = {}
    star_count = {}
    aspects_count = {k: {} for k in aspects}
    # for each review, count clue words in different aspects
    for _, row in movie.iterrows():
   
        director_names = ' '.join(director.Person[director.Movie_ID == row.Movie_ID].tolist()).split()
        star_names = ' '.join(star.Person[star.Movie_ID == row.Movie_ID].tolist()).split()
        for i, text in review.review_text[review.Movie_ID == row.Movie_ID].items():
            text = utils.unicodeToAscii(text)
            for w in text.split():
                w = w.strip(',.?!)')
                if w in director_names:
                    director_count[i] = director_count.get(i, 0) + 1
                if w in star_names:
                    star_count[i] = star_count.get(i, 0) + 1
                w = w.lower()
                for k in aspects_count:
                    if w in aspects[k]:
                        aspects_count[k][i] = aspects_count[k].get(i, 0) + 1
    
    review['director_count'] = pd.Series(director_count)
    review['star_count'] = pd.Series(star_count)                   
    

    df = movie.set_index('Movie_ID')
    df['director_content'] = (review.director_count > 0).groupby(review.Movie_ID).mean()
    df['star_content'] = (review.star_count > 0).groupby(review.Movie_ID).mean()
    for k in aspects_count:
        review[f'{k}_count'] = pd.Series(aspects_count[k])
        df[f'{k}_content'] = (review[f'{k}_count'] > 0).groupby(review.Movie_ID).mean()
    df.Rating =df.Rating.replace('None', 'Unrated').replace('Not Rated', 'Unrated')
    genre_set = []
    for g in df.Genre.apply(lambda x: x.split(',')):
        genre_set = g
    #genre_set.remove('Musical')
    genre_cols = []
    for g in genre_set:
        col = 'Genre_' + g.replace('-', '')
        genre_cols.append(col)
        df[col] = df.Genre.apply(lambda x: g in x)
    df.Release_Date = pd.to_datetime(df.Release_Date)
    df['Release_Month'] = df.Release_Date.dt.month
    df['Release_Year']=df.Release_Date.dt.year
    #2.4 Review Rating Classification
    review['rating_score'] = pd.to_numeric(review.rating.apply(lambda x: x.split('/')[0] if type(x) == str else  None))
    review_df = review[['review_text', 'rating_score']].dropna()
    # clean review text
    p = '[\d,]* out of [\d,]* found this helpful.*\nPermalink'
    review_df.review_text = review_df.review_text.apply(lambda x: re.sub(p, '', x))
    
    clf_rating = pickle.load(open('rating_classifier_model_new.pkl', 'rb'))
    review_to_predict = review[['review_text']][review.rating_score.isnull()]
    vect = pickle.load(open('vectorizer_new.pickle','rb'))
    X_to_predict = vect.transform(review_to_predict.review_text)
    review_to_predict['rating_score'] = clf_rating.predict(X_to_predict)
    predicted_rating = review.rating_score.fillna(review_to_predict.rating_score)
    df['avg_predicted_rating'] = predicted_rating.groupby(review.Movie_ID).mean()
    return df
#url="https://www.imdb.com/title/tt1093357/?ref_=nv_sr_srsg_0_tt_1_nm_0_q_tt1093357"

def final(df):
   
    #df_pipe= pd.get_dummies(df,columns=['Rating'])
    #df= df.rename(columns= {'Director_value':'director_value','Writer_value':'writer_value','Star_value':'star_value'})
    X = df.drop(['Release_Date', 'Director', 'Writer', 
                 'Star', 'Budget','Review_URL',
                'Title','Genre','Length','Rating', 'avg_predicted_rating'],axis=1)
    
    for col in ['Budget_Final', 'director_value', 'writer_value', 'star_value']:
        X[col] = X[col].apply(np.log1p)
    #X['Budget_Final'] = X["Budget_Final"].apply(np.log1p) 
    
   
    test_cols= X.columns.to_list()
    
    train_cols = ['Budget_Final', 'Release_Year', 'Release_Month', 'Length_Min',
                'review_count','director_value',
       'writer_value', 'star_value', 'acting_content', 'direction_content',
       'screenplay_content', 'sound_content', 'story_content',
       'visual_content', 'film_content', 
               'avg_review_length', 'avg_rating',
               'Genre_Biography', 'Genre_Thriller', 'Genre_Sport', 'Genre_Mystery',
               'Genre_Western', 'Genre_Action', 'Genre_SciFi', 'Genre_Crime',
               'Genre_Music', 'Genre_Drama', 'Genre_Family', 'Genre_Adventure',
               'Genre_Comedy', 'Genre_History', 'Genre_Romance', 'Genre_War',
               'Genre_Fantasy', 'Genre_Horror', 'Genre_Animation']
    # Get missing columns in the training test
    missing_cols = set( train_cols ) - set( test_cols )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X[c] = False
    # Ensure the order of column in the test set is in the same order than in train set
    X = X[train_cols]
    # X = df[cols]s
    #return X
    # # X = pd.get_dummies(X, prefix='Rating')
    clf_success= pickle.load(open('success_classifier_model_hgb.pkl', 'rb'))
    
    #result= clf_success.predict(X[train_cols])
    y_probs = clf_success.predict_proba(X[train_cols])
    #result = (y_probs >= 0.30).astype(int)
    # if result == 1:
    #     pred= 'Success's
    # else:
    #     pred= 'Fail'

    X1= df.drop(['Release_Year','Director', 'Writer', 
                 'Star', 'Budget','Review_URL',
                'Title','Genre'],axis=1)   
    test_cols= X1.columns.to_list()
    
    train_cols = ['Budget_Final', 'Release_Month', 'Length_Min',
       'director_value', 'writer_value', 'star_value', 'acting_content',
       'direction_content', 'screenplay_content', 'sound_content',
       'story_content', 'visual_content', 'film_content', 'review_count',
       'avg_review_length', 'avg_rating','avg_predicted_rating',
       'Genre_Biography', 'Genre_Thriller', 'Genre_Sport', 'Genre_Mystery',
       'Genre_Western', 'Genre_Action', 'Genre_SciFi', 'Genre_Crime',
       'Genre_Music', 'Genre_Drama', 'Genre_Family', 'Genre_Adventure',
       'Genre_Comedy', 'Genre_History', 'Genre_Romance', 'Genre_War',
       'Genre_Fantasy', 'Genre_Horror', 'Genre_Animation']
    # Get missing columns in the training test
    missing_cols = set( train_cols ) - set( test_cols )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X1[c] = False
    # Ensure the order of column in the test set is in the same order than in train set
    X1 = X1[train_cols] 
    X1.replace({False: 0, True: 1}, inplace=True)
    ss=MinMaxScaler() 
    # we are scaling the data for ANN. Without scaling it will give very poor results. Computations becomes easier
    X_scaled=pd.DataFrame(ss.fit_transform(X1),columns= X1.columns)
    hist_reg= pickle.load(open('success_regression_model_hgb.pkl', 'rb'))
    y_pred = hist_reg.predict(X_scaled) 
    ss= pickle.load(open('scaler.pkl', 'rb')) 
    y_pred= pd.DataFrame(y_pred)
    y_data = ss.inverse_transform(y_pred)
    return y_probs,y_data

option = st.selectbox(
    'Overall EDA or Single Movie',
    ('Overall EDA', 'Single Movie'))
st.write('You selected:', option)

if option =='Overall EDA':
    with open("eda.py") as f:
        exec(f.read())
if option=='Single Movie':  
    url = st.text_input("Enter movie url")
    if url:

        driver = webdriver.Chrome()
        df= preprocessing(url)
        result,result1 = final(df)
        st.dataframe(df)

        st.write("Success Probability is " + str(result[0][1]*100) +"%")
        st.write(result1)
        st.write("Recommended Movies based on your Movie Choice:")
        st.write(movie_recom.recommend(df.Title[0]))
        review_analysis.plots()








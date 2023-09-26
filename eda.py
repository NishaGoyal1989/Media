import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import re


movie = pd.read_csv('Sample Data/movie_preprocessed.csv')
review= pd.read_csv('review_preprocessed.csv')
movie.Rating = movie.Rating.replace('None', 'Unrated').replace('Not Rated', 'Unrated')
movie.dropna(subset=['Release_Date'],inplace = True)
movie.dropna(subset=['Writer'],inplace= True)
# convert release date into date type
movie.Release_Date = pd.to_datetime(movie.Release_Date.apply(lambda x: re.sub(' \(.*\)', '', x)), errors='coerce')

genre_set = set()
for g in movie.Genre.apply(lambda x: x.split(', ')).tolist():
    genre_set.update(g)
genre_set.remove('Musical')
genre_cols = []
for g in genre_set:
    col = 'Genre_' + g.replace('-', '')
    genre_cols.append(col)
    movie[col] = movie.Genre.apply(lambda x: g in x)
sorted_index = movie[genre_cols].sum().sort_values(ascending=False).index
genre_df = pd.DataFrame([movie[col].groupby(movie.Successful).sum() for col in genre_cols])

movie['Movie_ID'] = movie.Review_URL.apply(lambda x: x.split('/')[4])
crew = []
for i, r in movie.iterrows():
    movie_id = r.Movie_ID
    for director in r.Director.split('/'):
       
        crew.append([movie_id, director, 'director'])
    for writer in r.Writer.split('/'):
       
        crew.append([movie_id, writer, 'writer'])
    for star in r.Star.split('/'):
       
        crew.append([movie_id, star, 'star'])
person = pd.DataFrame(crew, columns=['Movie_ID', 'Person', 'Role'])
movie = movie.set_index('Movie_ID')
m_person = movie[['Successful']].merge(person, left_index=True, right_on='Movie_ID')
m_person.Person = m_person.Person.apply(lambda x: x.replace(' ', '_'))
text=[]
text.append( ' '.join(m_person.Person[m_person.Role == 'director'][m_person.Successful]))
text.append(' '.join(m_person.Person[m_person.Role == 'director'][~m_person.Successful]))
text.append( ' '.join(m_person.Person[m_person.Role == 'writer'][m_person.Successful]))
text.append(' '.join(m_person.Person[m_person.Role == 'writer'][~m_person.Successful]))
text.append( ' '.join(m_person.Person[m_person.Role == 'star'][m_person.Successful]))
text.append(' '.join(m_person.Person[m_person.Role == 'star'][~m_person.Successful]))

def wordcloud():
    fig, ax = plt.subplots(1,2,figsize=(50, 40))
    
    for t in range(6):
        i=t+1
        plt.subplot(3,2,i).set_title("Topic #" + str(t))
        plt.plot()
        plt.imshow(WordCloud(max_font_size=60).generate(text[t]),interpolation='bilinear')
        plt.axis("off")
    st.pyplot(fig)
def countplot():
    fig, ax = plt.subplots(2,3,figsize=(50,30))
    plt.xticks(rotation = 90)
    sns.countplot(x=movie.Rating,order = movie.Rating.value_counts().index,ax=ax[0,0])
    sns.countplot(x=movie.Rating,hue= movie.Successful,order = movie.Rating.value_counts().index,ax=ax[0,1])
    sns.countplot(x=movie.Release_Date.dt.month,hue= movie.Successful,ax=ax[0,2])
    sns.countplot(x=movie.Release_Year,hue= movie.Successful,ax=ax[1,0])
    genre_df.loc[sorted_index].plot.bar(ax=ax[1,1])
    (genre_df[True] / (genre_df[True] + genre_df[False])).loc[sorted_index].plot.bar(ax=ax[1,2])
    st.pyplot(fig)

option = st.selectbox(
    'What you want to visualize?',
    ('Countplot', 'Wordcloud', 'Inference Report'))
st.write('You selected:', option)

if option == 'Countplot':
    countplot()
if option =='Wordcloud':
    wordcloud()
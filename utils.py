import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium import webdriver
import re
import unicodedata
import time

def get_movie_info(movie_url):
    HEADERS ={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    response = requests.get(movie_url,headers=HEADERS)
    soup = BeautifulSoup(response.text, 'lxml')
    title = soup.find_all(class_='sc-afe43def-1 fDTGTb')[0].get_text()
    rating = soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers sc-afe43def-4 kdXikI baseAlt')[0].find_all('li')[1].get_text() 
    length = soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers sc-afe43def-4 kdXikI baseAlt')[0].find_all('li')[2].get_text()
    release_date= soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content base')[3].find_all('li')[0].get_text()
    genres = soup.find_all(class_='ipc-chip-list__scroller')[0].find_all('a')
    list1=[]
    for genre in genres:
        
        list1.append(genre.get_text())
    list1= ','.join(list1)
    director=[] 
    writer=[]
    star=[]
    directors=soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')[0]
    writers=soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')[1]
    stars=soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')[2]
    review_url= 'https://www.imdb.com'+soup.find_all(class_='ipc-link ipc-link--baseAlt ipc-link--touch-target sc-9e83797f-2 jJzPWH isReview')[0]['href']
    budget= soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content base')[10].find_all('li')[0].get_text()
    #box_office = soup.find_all(class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content base')[13].find_all('li')[0].get_text()
    for item in directors:
        director.append(item.get_text())
    director='/'.join(director)
    for item in writers:
        writer.append(item.get_text())
    writer='/'.join(writer)
    for item in stars:
        star.append(item.get_text())
    star='/'.join(star)            
    return (title,rating,list1,length,release_date,director,writer,star,budget,review_url)

def get_movie_id(review_url):
    return review_url.split('/')[4]

def get_review_info(review):
    #driver = webdriver.Chrome()
    try:
        rating = review.find_element(By.CSS_SELECTOR,'span.rating-other-user-rating').text
    except:
        rating = None

    try:
        title = review.find_element(By.CSS_SELECTOR,'a.title').text
    except:
        title = None

    try:
        user_name = review.find_element(By.CSS_SELECTOR,'span.display-name-link').text
    except:
        user_name = None

    try:
        user_url = review.find_element(By.CSS_SELECTOR,'span.display-name-link a').get_attribute('href')
    except:
        user_url = None

    try:
        review_date = review.find_element(By.CSS_SELECTOR,'span.review-date').text
    except:
        review_date = None
    try:
        helpful = review.find_element(By.CSS_SELECTOR,'div.text-muted').text
    except:
        helpful = None

    try:
        expander = review.find_element(By.CSS_SELECTOR,'div.spoiler-warning__control')
        expander.click()
    except:
        expander = None

    try:
        content = review.find_element(By.CSS_SELECTOR,'div.content').text
    except:
        content = None
        
    return [rating, title, user_name, user_url, review_date, helpful, expander is not None, content]

def load_page(driver):
    i=0
    while i<5:
        try:
            loadMoreButton = driver.find_element(By.ID,"load-more-trigger")
            time.sleep(2)
            loadMoreButton.click()
            time.sleep(2)
        except Exception as e:
            print('e')
            break

# Convert length to min
def parse_time(x):
    if not x:
        return x
    length = 0
    h = re.search('(\d*)h', x)
    
    if h:
        length += int(h.group(1)) * 60

    m = re.search('(\d*)m', x)
    
    if m:
        length += int(m.group(1))
    return length

def unicodeToAscii(s):
        
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')


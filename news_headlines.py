import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests import get
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import datetime
import time
from dateutil import parser
import json
import os
from selenium.webdriver.chrome.service import Service


nltk.downloader.download('vader_lexicon')
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("—disable-gpu")
driver=webdriver.Chrome(options=chrome_options,executable_path='./chromedriver')
driver.get("https://finance.yahoo.com/topic/crypto/")
sentiment={}
sentiment['crypto']=[]
yahoo={}
sentiment['crypto'].append(yahoo)
while True:
    try:
        driver.find_element(By.XPATH, '//*[@id="Fin-Stream"]/ul/li[150]')
        break
    except:
        driver.find_element(By.XPATH, '//*[@id="atomic"]/body').send_keys(Keys.END)
for nb in range(1, 151):
    article_content=''
    unix_time=''
    vader=SentimentIntensityAnalyzer()
    title=driver.find_element(By.XPATH, '//*[@id="Fin-Stream"]/ul/li['+str(nb)+']/div/div/div[2]/h3/a').text
    link=driver.find_element(By.XPATH, '//*[@id="Fin-Stream"]/ul/li[' + str(nb) + ']/div/div/div[2]/h3/a').get_attribute('href')
    req=requests.get(link)
    page=req.content
    soup=BeautifulSoup(page,'html.parser')
    for paragraphs in soup.find_all('div', class_="caas-body"):
        article_content+=paragraphs.text
    for d in soup.findAll('time'):
        if d.has_attr('datetime'):
            date_time=parser.parse(str(d['datetime']))
            unix_time=date_time.timestamp()
    score=vader.polarity_scores(article_content)
    yahoo['article'+' - '+str(unix_time)]={'title':title, 'content':article_content,'time':unix_time, 'ntlk_sentiment':score}

print(yahoo)

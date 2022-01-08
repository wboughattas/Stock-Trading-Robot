import pandas as pd
from bs4 import BeautifulSoup
import nltk
from requests import get
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

driver=webdriver.Chrome('../../Desktop/Stock-Trading-Robot/chromedriver')
driver.get("https://finance.yahoo.com/topic/crypto/")
news_headlines=[]

while True:
    try:
        driver.find_element(By.XPATH, '//*[@id="Fin-Stream"]/ul/li[150]')
        break
    except:
        driver.find_element(By.XPATH, '//*[@id="atomic"]/body').send_keys(Keys.END)
for nb in range(1, 151):
    news_headlines.append(driver.find_element(By.XPATH, '//*[@id="Fin-Stream"]/ul/li['+str(nb)+']/div/div/div[2]/h3/a').text)
print(news_headlines)

# BloombergBtc="https://www.bloomberg.com/search?query=bitcoin"
# responseBloom=get(BloombergBtc)
# soup2=BeautifulSoup(responseBloom.text,'html.parser')
# for div in soup2.findAll('div', attrs={'class':'storyItem__'}):
#     for bloomheadlines in div.find_all('a', attrs={'class':'headline__'}):
#         new_headlines.append(bloomheadlines.text)
#         print(bloomheadlines)





import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

def get_hlinks(samples):
    hrefs = []
    for s in samples:
        ss = s.find('a').attrs['href']
        if ss.find('wordlist/letter') > -1:
            hrefs.append(ss)
    return hrefs
 
result = requests.get("https://www.bgjargon.com/wordlist/letter/%D0%90")
c = result.content
soup = BeautifulSoup(c)
samples = soup.find_all('li')
hlinks = get_hlinks(samples)

data = []
for hlink in hlinks:
    content = requests.get(hlink).content
    soup = BeautifulSoup(content)
    samples = soup.find_all('li')
    for s in samples:
        ss = s.find('a')
        if ss.attrs['href'].find('meaning') > -1:
            z = s.find('a')
            data.append(z.contents)

# save to disk
bgjargon = pd.DataFrame(data, columns=['phrase'])
bgjargon.to_csv('../data/bgjargon.csv', encoding='utf-8')

# read them from disk
bgjargon_pd = pd.read_csv("../data/bgjargon.csv", )[['phrase']]
# bgjargon_pd
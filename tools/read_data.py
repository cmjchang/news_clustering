import re

import pandas as pd
from bs4 import BeautifulSoup


def read_from_file(file_list):
    dfs = []
    for file in file_list:
        with open(file, 'rb') as f:
            soup = BeautifulSoup(f.read().decode('utf8'), features="html.parser")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'), articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                # replace date
                cleanr = re.compile(
                    u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                data = {"heading": heading, "content": content}
                data = pd.DataFrame(data, index=[0])
                dfs.append(data)
    google_df = pd.concat(dfs, ignore_index=True)
    return google_df

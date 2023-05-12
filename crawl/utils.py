from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor
from news import NateNews
from typing import List, Union

import datetime as dt
import pandas as pd
import requests
import time


LINK = 'https://news.nate.com/view/'
CATEGORY=['eco', 'soc', 'int', 'its']

def get_news_df(
    news_list: List[NateNews]
):
    """make `pd.DataFrame` with `news_list`

    Returns:
        pd.DataFame: DataFrmae w.r.t `news_list`
    """
    info_list = list()
    for news in news_list:
        try:
            info_list.append(news.get_dict())
        except:
            print('Error occurs', ends=' ')
            print(news.url)

    return pd.DataFrame(info_list)


def get_news(
    url_list: Union[List[str], str]
):
    """Return `NateNews` list

    Args:
        url_list (Union[List[str], str]): url list to be requested

    Returns:
        List[Union[Response, None]]:
            1. NateNews: Normal Request
            2. None: Abnormal Request(won't get that page)
    """
    url_list = url_list if isinstance(url_list, list) else [url_list]

    if len(url_list) < 100:
        with ThreadPoolExecutor(max_workers=10) as mult:
            _news_list = list(mult.map(_create, url_list))
    else:
        _news_list = list()
        for url in url_list:
            time.sleep(0.5)
            _news_list.append(_create(url))
    
    news_list = [news for news in _news_list if news]
    return news_list

def get_ranking(
    date1: Union[int, None]=None,
    date2: Union[int, None]=None,
):
    """Return articles of ranking from `date1` to `date2`
    
    Args:
        date1 (Union[int, None], optional):
            None -> `datetime.datetime.now()`
        date2 (Union[int, None], optional): 
            None -> `date1`
    
    Returns:
        List[str]: url list of news from date1 to date2
    """
    date1 = date1 if date1 else dt.date.today().strftime('%Y%m%d')
    date2 = date2 if date2 else date1
    
    date1 = int(date1)
    date2 = int(date2)
    
    date_list = _get_date_list(date1, date2)
    
    ranking_url_list = [_ranking_url(date, news_category) for date in date_list for news_category in CATEGORY]
    ranking_url_list = sum(ranking_url_list , [])
    ranking_url_list = list(set(map(lambda x: f"https:{x[:35]}", ranking_url_list)))
    
    # Sorting
    return sorted(ranking_url_list)

def _ranking_url(date:int, category:str):
    # to prevent request error
    print(date, category)
    time.sleep(0.1)
    req = requests.get(f"https://news.nate.com/rank/interest?sc={category}&p=day&date={date}")
    soup = bs(req.text, 'html.parser')
    main = soup.find('div', {'class': 'postRankNews'})
    links = main.find_all('a')
    return [link['href'] for link in links if 'nate.com/view' in link['href']]

def get_urls(
    date1: Union[int, None]=None,
    date2: Union[int, None]=None,
    artc1: Union[int, None]=None,
    artc2: Union[int, None]=None,
):
    """get url list
    
    Desc:
        url list
        eg:
            `date1`: `article1` ~ `article2`
            `dateN`: `article1` ~ `article2`
            \n\t\t...
            `date2`: `article1` ~ `article2`

    Args:
        date1 (Union[int, None], optional):
            None -> `datetime.datetime.now()`
        date2 (Union[int, None], optional): 
            None -> `date1`
        artc1 (Union[int, None], optional):
            None -> 1: first article of that day
        artc2 (Union[int, None], optional):
            None -> last article of that day

    Returns:
        List[str]: url list
    """    
    date1 = date1 if date1 else int(dt.datetime.now().strftime('%Y%m%d'))
    date2 = date2 if date2 else date1
    artc1 = artc1 if artc1 and artc1 > 0 else 1

    urls = [
        f"{LINK}{date}n{str(num).zfill(5)}"
        for date in _get_date_list(date1, date2)
        for num in _get_artc_list(artc1, artc2, date)
    ]
    return urls


def _get_date_list(
    date1: int,
    date2: int, 
):
    """get date list

    Args:
        `date1` (int): first date
        `date2` (int): last date

    Returns:
        List[int]: date list from `date1` to `date2`
    """
    
    if not date2:
        return [date1]
    
    date:int = date1
    date_list = list()
    
    while date <= date2:
        date_list.append(date)
        date = dt.datetime.strptime(str(date), "%Y%m%d") + dt.timedelta(days=1)
        date = int(date.strftime('%Y%m%d'))
    
    return date_list


def _get_artc_list(
    artc1: int,
    artc2: Union[int,None],
    date: int,
):
    """get article list

    Args:
        `artc1` (int): first article in nate news
        `artc2` (Union[int,None]): last article in nate news
            None: `artc2` = latest article on `date`
        `date` (int): the day you want to crawl

    Returns:
        List[int]: article list from `artc1` to `artc2`
    """
    max_article = _get_recent(date)
    artc1 = artc1 if artc1 < max_article else max_article

    if not artc2 or artc2 > max_article:
        artc2 = max_article
    return [artc for artc in range(artc1, artc2+1)]


def _get_recent(date: int):
    """get latest article number in Nate given date

    Args:
        `date` (int): date in which latest article number will be found

    Note:
        Can't return accurate number of article
        -> get latest number of article in '최신뉴스' in Nate

    Returns:
        int: latest article number
    """        
    req = requests.get(f'https://news.nate.com/recent?mid=n0100&type=c&date={date}')
    content = bs(req.text, 'html.parser')
    _recent = content.find_all('div', {'class': 'mlt01'})
    
    latest = None
    for news in _recent:
        # recent = //news.nate.com/view/{YYYY}{mm}{dd}n{NNNNN}?mid=n0100
        recent = int(news.find('a')['href'].split('?')[0][-5:])
        if not latest or latest < recent:
            latest = recent
    return latest # return latest article number


def _create(url:str):
    """create `NateNews` if it satisfy some conditions

    Args:
        `url` (str): url for news in Nate

    Desc:
        return `NateNews` if given url satisfy some conditions
        * 1. Should have article(articleContetns)
        * 2. Exclude English news
        
    Returns:
        Union[NateNews, None]: 
    """        
    # time.sleep(0.5)
    # TODO: handling sleep stuff... => Only when collect huge amount of dataset
    news = NateNews(url)
    
    # 연예 기사는 제외
    
    if news.category in [
        "연예가화제",
        "방송/가요",
        "영화",
        "해외연예",
        "POLL",
        "포토/TV",
        "아이돌24시"
    ]:
        print(f"{news.url} is Entertainment News!")
        return None

    # 연합 뉴스들 제외
    if news.press == 'AP연합뉴스' or news.press == 'EPA연합뉴스':
        print(f"{news.url} is English News!")
        return None
    
    # 기사가 없는 경우
    if not news.text:
        print(f"{news.url} has no article!")
        return None
    else:
        # 특수 기사들은 제외
        if '[속보]' in news.title or\
            '[포토]' in news.title or\
            '[부고]' in news.title or\
            '[인터뷰]' in news.title:
            print(f"{news.url} is not Normal News!")
            return None

    if "[NO RELATION]" in news.text:
        print(f'{news.url} has a low-relevant image!')
        return None

    # 기사가 있다 -> 길이 확인하기, 길이가 짧을 시에 제외
    if len(news.text) < 20:
        print(f'{news.url} has too short article!')
        return None
    else:
        print(f'{news.url}')
        return news
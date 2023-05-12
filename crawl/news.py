import requests

from bs4 import BeautifulSoup as bs
from preprocessing import text_cleaning


LINK = 'https://news.nate.com/view/'
HEADERS = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}


class NateNews:
    def __init__(self, url:str):
        self.url = _convert_url(url)
        res = requests.get(self.url, headers=HEADERS) # to prevent block crawling
        assert res.status_code == 200 # to check valid request
        
        self.content = bs(res.text, 'html.parser')
        self.id = self.url.split('/')[-1].replace('n', '')
        self.title = self._get_title()
        self.category = self._get_category()
        self.press = self._get_press()
        self.date = self._get_date()
        self.text = self._get_text()
    
    # for forwarding to flask
    def get_dict(self):
        return{
            "title": self.title,
            "category": self.category,
            "press": self.press,
            "date": self.date,
            "content": self.text,
            "image": self.image,
            "url": self.url
        }

    # for json file
    def get_json(self):
        return{
            "id": self.id,
            "category": self.category,
            "text": f"{self.title}[SEP]{self.text}",
            "img": f"{self.id}.png",
            "img_url": self.image,
            "url": self.url
        }
    
    ...
    
    def _get_title(self):
        _title = self.content.find('h3', {'class': 'viewTite'})
        if not _title:
            _title = self.content.find('h3', {'class': 'articleSubecjt'})
        try:
            title = _title.text
        except:
            title = ''
        return title

    def _get_category(self):
        try:
            nav = self.content.find('div', {'class': 'snbArea'})
            _category = nav.find('li', {'class': 'on'})
            category = _category.text
        except:
            category = ''
        return category
    
    def _get_press(self):
        _press = self.content.find('a', {'class': 'medium'})
        try:
            if _press and _press.text:
                press = _press.text
            else:
                press = self.content.find('dl', {'class': 'articleInfo'}).select('img')[0]['alt']
        except:
            press = ''
        return press
    
    def _get_date(self):
        try:
            date = self.content.find('em').text
        except:
            date = ''
        return date
    
    def _get_text(self):
        _text = self.content.find('div',{'id': 'articleContetns'})
        try:
            article, self.image = text_cleaning(_text)
        except:
            article, self.image = '', ''
        # article, self.image = text_cleaning(_text)
        return article

    ...

    # for content
    @property
    def content(self):
        return self._content
    
    @content.setter
    def content(self, article):
        self._content = article
    

    # for title
    @property
    def title(self):
        return self._title
    
    @title.setter
    def title(self, title: str):
        self._title = title
    
    
    # for category
    @property
    def category(self):
        return self._category
    
    @category.setter
    def category(self, category: str):
        self._category = category

    
    # for press
    @property
    def press(self):
        return self._press
    
    @press.setter
    def press(self, press: str):
        self._press = press
        
    
    # date
    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date: str):
        self._date = date

    
    # text
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, text: str):
        self._text = text

    
    # image
    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, img_url: str):
        self._image: str = img_url


    # url
    @property
    def url(self):
        return self._url
    
    @url.setter
    def url(self, url: str):
        self._url = url


def _convert_url(url:str):
    url = url.split('?')[0].split('//')[-1]
    return f"https://{url}"
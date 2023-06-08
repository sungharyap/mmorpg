import bs4
import re

from bs4 import BeautifulSoup as bs

# 크롤 중 아래 메시지 뜨는 경우 있음. 없애기 위해 아래의 원문 추가
MESSAGE = '지원하지 않는 브라우저로 접근하셨습니다.\nInternet Explorer 10 이상으로 업데이트 해주시거나, 최신 버전의 Chrome에서 정상적으로 이용이 가능합니다.'

# 언론사별 불용어 지정
PRESS = [
    # 중앙일보
    '중앙일보, 무단 전재 및 재배포 금지',
    # 노컷뉴스
    'CBS노컷뉴스는 여러분의 제보로 함께 세상을 바꿉니다. 각종 비리와 부당대우, 사건사고와 미담 등 모든 얘깃거리를 알려주세요.이메일 : 카카오톡 : @노컷뉴스사이트 : https://url.kr/b71afn',
    # 뉴스1
    'news1.kr',
    'reserved.',
    '무단 전재 및 재배포 금지.',
    '뉴스1.',
    # 뉴시스
    '뉴시스통신사.',
    '무단전재-재배포 금지.',
    # 데일리임팩트
    '데일리임팩트.',
    # 조선비즈
    'ChosunBiz.com',
    # 헤럴드경제, 헤럴드 POP
    'Reserved.',
    # 더팩트
    '여러분의 제보를 기다립니다.',
    # 블로터
    'bloter.net',
    # 케이스타뉴스
    '케이스타뉴스.',
    # JTBC
    'JTBC의 모든 콘텐트는 저작권법의 보호를 받은바, 무단 전재, 복사, 배포 등을 금합니다.',
    # SBS
    'SBS & SBS Digital News Lab.',
    # 에이빙뉴스
    '에이빙.',
    # 뉴스엔
    '뉴스엔.',
    # 마이데일리
    '마이데일리.',
    # 스포츠한국
    '스포츠한국. 무단전재 및 재배포 금지',
    # 스포탈코리아
    '스포탈코리아. 무단전재 및 재배포 금지',
    # 인터풋볼
    '인터풋볼. 무단전재 및 재배포 금지',
    # 일간스포츠
    '일간스포츠. All rights reserved',
]

def text_cleaning(article: bs4.element.Tag):
    """Cleaning news content and divide images & texts

    Args:
        article (bs4.element.Tag): content of article found with `bs`

    Returns:
        Tuple[str, Dict[str, str]]:
            str: Content of news article after cleaning noise
            Dict[str, str]: pair of image-caption
    """    
    article_text = str(article)
    # img alt 모두 제거
    pattern = "alt=\"[^\"]+\""
    tmp = re.sub(pattern, '', article_text)
    
    # <b> 제거
    pattern = '<b>[^<]*</b>'
    tmp = re.sub(pattern, '', tmp)
    
    # <i> 제거
    pattern = '<i>[^<]*</i>'
    tmp = re.sub(pattern, '', tmp)
    
    # <strong> 제거
    pattern = '<strong>[^<]*</strong>'
    tmp = re.sub(pattern, '', tmp)
    
    # [] 내부 모두 제거
    pattern = '\[[^\]]*\]'
    tmp = re.sub(pattern, '', tmp)
    
    tmp = tmp.replace('\n', '').replace('\t', '').replace('\r', '') # 공백 제거
    pattern = "<br/?>" # <br> 태그 -> 개행으로 변경
    tmp = re.sub(pattern, '[NEWLINE]', tmp)
    
    tmp = _remove_caption(tmp)
    tmp = _remove_html_tag(tmp)

    content = bs(tmp, 'html.parser') # 다시 parsing
    tmp = re.sub(' {2,}', ' ', content.text)
    tmp = tmp.replace("[NEWLINE]", "\n")

    tmp = _remove_bracket(tmp)
    
    tmp = tmp.replace('·', ', ')
    tmp = ('').join([word for word in tmp if word.isalpha() or ord(word) < 128 or word == '…'])
    tmp = tmp.replace(MESSAGE, '')    
    
    tmp = _remove_email(tmp)

    text = tmp.replace('기사내용 요약', '[기사내용 요약]\n')
    text = re.sub("\\'", "", text)
    
    text = _remove_press(text)
    
    # text -> article, images -> {IMAGE: CAPTION}
    text, img_url = _seperate_text(text)
    text = _remove_link(text)
    text = _remove_newline(text)
    
    text = ('.').join(text.split('.')[:-1])
    text = re.sub('\n{3,}', '\n\n', text)

    text = "[NO RELATION]" if re.search('사진.{0,4}기사.{0,12}관련.{0,5}없|사진.{0,4}기사.{0,6}연관.{0,5}없', text) else text

    return (text.strip() + '.', img_url)

def _seperate_text(text):
    """Seperate text with images(and captions)
    """
    pattern_link = re.compile('\[(http://[^\]]*)\]')
    result_img = pattern_link.finditer(text)
    text = re.sub(pattern_link, '', text)

    pattern_cap = re.compile('\[([^\]]*)\]')
    result_cap = pattern_cap.finditer(text)
    text = re.sub(pattern_cap, '', text)
    text = re.sub('\n{3,}', '\n\n', text)
    
    img_url = ""
    for r in result_img:
        img_url = r.group(1)
        try:
            caption:str = next(result_cap).group(1)
            if re.search('사진.{0,4}기사.{0,12}관련.{0,5}없|사진.{0,4}기사.{0,6}연관.{0,5}없', caption):
            # text remove
                text = "[NO RELATION]"
        except:
            pass
        break
        # try:
        #     image_dict[r.group(1)] = next(result_cap).group(1)
        # except:
        #     pass
    text = text if img_url else ""
    return text, img_url

def _remove_press(text):
    """Remove noise in each article in press
    """
    text = re.sub(('|').join(PRESS), '', text)
    return text

def _remove_link(text):
    """Remove links
    """
    pattern = re.compile('^[^\[\n]*https?://[^ \n]+', re.MULTILINE)
    text = re.sub(pattern, '', text)
    return text

def _remove_caption(text):
    """Specify which line is caption of certain image
    """
    pattern = re.compile('(<p style="[^>]*>)([^<]*)(</p>)')
    result = pattern.finditer(text)
    for r in result:
        text = text.replace(r.group(), f"[{r.group(2)}]")
    # 캡션 표시
    pattern = re.compile('(<span class="sub_tit">)([^<]*)(</span>)')
    result = pattern.finditer(text)

    for r in result:
        text = text.replace(r.group(), f"[{r.group(2)}]")
    
    return text

def _remove_html_tag(text):
    """Replace html tags with newline or empty string
    """
    pattern = "</?p[^>]*>" # <p> or </p> -> 개행으로 변경
    text = re.sub(pattern, '\n', text)
    pattern = "<caption>[^>]+>" # caption 제거
    text = re.sub(pattern, '', text)
    pattern = "<a[^<]+</a>" # [a] 태그 제거
    text = re.sub(pattern, '', text)

    # pattern = "<img[^>]+>" # img들 모두 제거
    pattern = re.compile('<img[^>]+>')
    result = pattern.finditer(text)
    images = bs(text, 'html.parser').find_all('img')
    
    i = 0
    for r in result:
        text = text.replace(r.group(), f"\n[http:{images[i]['src']}]\n")
        i += 1
    
    return text

def _remove_bracket(text):
    """Remove <> and [] stuffs
    """
    pattern = "<[^>]*>" # <> 내부 모두 제거
    text = re.sub(pattern, '', text)
    
    pattern = "\([^\)]*\)" # () 내부 모두 제거
    text = re.sub(pattern, '', text)
    
    return text

def _remove_email(text):
    """Remove texts with email form
    """
    pattern = '[a-zA-Z0-9+-_.]+@[a-zA-Z0-9+-_]+.com'
    text = re.sub(pattern, '', text)
    pattern = '[a-zA-Z0-9+-_.]+@[a-zA-Z0-9+-_]+.co.kr'
    text = re.sub(pattern, '', text)
    pattern = '[a-zA-Z0-9+-_.]+@'
    text = re.sub(pattern, '', text)
    
    return text

def _remove_newline(text):
    """Remove newline and similar things
    """
    text = re.sub('\n ', '\n\n', text)
    text = re.sub('-\n', '\n\n', text)
    text = re.sub('\n{2,}', '\n\n', text)
    text = re.sub('- \n', '\n\n', text)
    text = re.sub('\n-', '\n\n', text)

    text = text.strip()
    if len(text) < 2:
        return ''
    if text[0] == ']': text = text[1:]
    return text
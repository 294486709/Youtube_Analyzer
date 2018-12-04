from bs4 import BeautifulSoup
import requests
import os
from urllib import request
import split_sentence


os.makedirs('./articles', exist_ok=True)

class Crawl_CNN:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_begin = soup.find_all('p', {'class': 'zn-body__paragraph speakable'})
        for m in content_begin:
            with open('./articles/%s' % article_name, 'a') as f1:
                f1.write(m.get_text())

        content_body = soup.find_all('div', {'class': 'zn-body__paragraph'})
        for i in content_body:
            with open('./articles/%s' % article_name, 'a') as f2:
                f2.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_BBC:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('div', {'class': 'story-body__inner'})
        for i in content_body:
            j = i.find_all('p')
            for k in j:
                with open('./articles/%s' % article_name, 'a') as f1:
                    f1.write(k.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_NYtimes:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('p', {'class': 'css-1ygdjhk e2kc3sl0'})
        for m in content_body:
            with open('./articles/%s' % article_name, 'a') as f1:
                f1.write(m.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_Time:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('div', {'class': 'padded'})
        for m in content_body:
            n = m.find_all('p')
            for i in n:
                with open('./articles/%s' % article_name, 'a') as f:
                    f.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_Newsweek:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15;"
        headers = {'user-Agent': user_agent}
        req = request.Request(self.url, headers=headers)
        html = request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')


        # write body
        content_body = soup.find_all('div', {'class': 'article-body'})
        for m in content_body:
            n = m.find_all('p')
            for i in n:
                with open('./articles/%s' % article_name, 'a') as f1:
                    f1.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

def main(url):

    url_cnn = 'https://www.cnn.com'
    url_bbc = 'https://www.bbc.com'
    url_nytimes = 'https://www.nytimes.com'
    url_time = 'http://time.com'
    url_newsweek = 'https://www.newsweek.com'

    if url_cnn in url:
        write_to_file = Crawl_CNN(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_bbc in url:
        write_to_file = Crawl_BBC(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_nytimes in url:
        write_to_file = Crawl_NYtimes(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_time in url:
        write_to_file = Crawl_Time(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_newsweek in url:
        write_to_file = Crawl_Newsweek(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    else:
        print('crawl is not available on this website')




if __name__ == '__main__':
    while True:
        url = input('please input url\n')
        if url == 'break':
            print('finish')
            break
        else:
            a = split_sentence.split_sentence(main(url))
            print(a)
            pass



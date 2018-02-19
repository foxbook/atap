import bs4
import requests

sources = ['https://www.washingtonpost.com',
            'http://www.nytimes.com/',
            'http://www.chicagotribune.com/',
            'http://www.bostonherald.com/',
            'http://www.sfchronicle.com/']

def crawl(url):
    domain = url.split("//www.")[-1].split("/")[0]
    html = requests.get(url).content
    soup = bs4.BeautifulSoup(html, "lxml")
    links = set(soup.find_all('a', href=True))
    for link in links:
        sub_url = link['href']
        page_name = link.string
        if domain in sub_url:
            try:
                page = requests.get(sub_url).content
                filename = "-".join(page_name.split()).lower() + '.html'
                with open(filename, 'wb') as f:
                    f.write(page)
            except:
                pass

if __name__ == '__main__':
    for url in sources:
        crawl(url)
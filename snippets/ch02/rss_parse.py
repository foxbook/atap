import bs4
import feedparser

feeds = ['http://blog.districtdatalabs.com/feed',
         'http://feeds.feedburner.com/oreilly/radar/atom',
         'http://blog.kaggle.com/feed/',
         'http://blog.revolutionanalytics.com/atom.xml']

def rss_parse(feed):
    parsed = feedparser.parse(feed)
    posts = parsed.entries
    for post in posts:
        html = post.content[0].get('value')
        soup = bs4.BeautifulSoup(html, 'lxml')
        post_title = post.title
        filename = "-".join(post_title.split()).lower() + '.xml'
        TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']
        for tag in soup.find_all(TAGS):
            paragraphs = tag.get_text()
            with open(filename, 'a') as f:
                f.write(paragraphs + '\n \n')
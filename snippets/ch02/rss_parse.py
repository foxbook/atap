import feedparser
import bs4
from slugify import slugify
from multiprocessing.dummy import Pool

feeds = ['http://blog.districtdatalabs.com/feed',
         'http://feeds.feedburner.com/oreilly/radar/atom',
         'http://blog.kaggle.com/feed/',
         'http://blog.revolutionanalytics.com/atom.xml']

def rss_parse(feed):
    parsed = feedparser.parse(feed)
    posts = parsed.entries
    for post in posts:
        html = post.content[0].get('value')
        soup = bs4.BeautifulSoup(html, "lxml")
        post_title = post.title
        filename = slugify(post_title).lower() + '.txt'
        TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']
        for tag in soup.find_all(TAGS):
            paragraph = tag.get_text()
            with open(filename, 'a') as f:
                f.write(paragraph + "\n \n")

def multi_proc_rss(feed_list, threads=1):
    pool = Pool(threads)
    pool.map(rss_parse, feed_list)
    pool.close()
    pool.join()

multi_proc_rss(feeds, 4)

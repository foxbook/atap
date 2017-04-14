import bs4

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']

def read_html(path):
    with open(path, 'r') as f:
        html = f.read()
        soup = bs4.BeautifulSoup(html, "lxml")
        for tag in soup.find_all(TAGS):
            yield tag.get_text()

filename = 'five-myths-about-gymnastics.html'

for paragraph in read_html(filename):
    print(paragraph + "\n")

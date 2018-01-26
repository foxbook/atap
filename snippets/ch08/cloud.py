import json
import codecs
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud, STOPWORDS

if __name__ == '__main__':
    # File contains a list of characters, reverse sorted by frequency
    # And a dict with {chapter title: chapter text} key-value pairs
    with codecs.open('data/oz.json', 'r', 'utf-8-sig') as data:
        text = json.load(data)
        oz_text = ''.join(chapter for chapter in text['chapters'].values())

        road_mask = np.array(Image.open("figures/spiral.jpg"))
        stopwords = set(STOPWORDS)

        # Generate a word cloud image
        wc = WordCloud(background_color="white", max_words=2000,
                       mask=road_mask, max_font_size=80,
                       stopwords=stopwords, colormap="autumn")

        wordcloud = wc.generate(oz_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
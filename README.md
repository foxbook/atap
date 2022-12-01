# Applied Text Analysis with Python

Welcome to the GitHub repo for our book!

In this book, we explore machine learning for text analysis as it relates to the data product pipeline. We discuss data ingestion and wrangling to preprocess input text into a corpus that can be analyzed. We explore methods for reading and analyzing corpora to build models that can be used in data applications, while monitoring them for change. Finally we discuss how to begin operationalizing these methods, moving towards a complete picture of how to build language aware data products.

## Using the Code

At this point the code in the book is over 5 years old, which in tech terms is ancient -- the code was written using Python 3.6 and Python 3.7 and the dependencies in the book such as spaCy and NLTK have gone through major version revisions, changing their APIs. We believe that the code in the book is still useful as a reference to read and understand the mechanics behind operationalizing applied text anlaysis but it cannot serve as a useful library without large structural changes and frequent maintenace to keep it up to date with its dependencies. To strike a balance between understanding and long term maintenance we've marked this repository as read only and closed the issues. 

We encourage you to try to implement the code in the book for yourself from scratch, as that is one of our best techniques for learning the interactions between the various methods and components in code! You'll likely get exceptions from the dependencies whose APIs have changed - but remember, those exceptions are there to help you and to communicate where things have gone wrong. With a little patience, you can work through those issues and hopefully develop a library that fits your specific needs while gaining practical experience and the pride of your accomplishments. 

Thank you again for reading Applied Text Analysis with Python and for being a part of our open source community. 

## Corpora and Copyright

Much of the code in this book is based around a large corpus of ingested RSS feeds. You are encouraged to construct your own sample using the tools detailed in Chapter 2: Building a Custom Corpus. However, we have also made available a [random 515.9 MB sample](https://bit.ly/3wa7kiK) of 33,232 files (~5%) of the full data source. This sample is post-processed. You can obtain a 7.2 GB raw sample [here](https://bit.ly/3wmVTU1). 

A note on the copyright of the corpus - this corpus is intended to be used for academic purposes only, and we expect you to use them as though you downloaded them from the RSS feeds yourself. What does that mean? Well it means that the copyright of each individual document is the copyright of the owner who gives you the ability to download a copy for yourself for reading/analysis etc. We expect that you'll respect the copyright and not republish this corpus or use it for anything other than tutorial analysis.

## Other Data Sets

Below are links to the other data sets used throughout the book (in order of appearance). 

- [NY Times Article in Chapter 1](https://www.nytimes.com/2017/01/26/arts/dance/rehearse-ice-feet-repeat-the-life-of-a-new-york-city-ballet-corps-dancer.html)
- [Pitchfork Music Album Reviews in Chapters 2 and 12](https://www.kaggle.com/nolanbconaway/pitchfork-data/data)
- [Wizard of Oz in Chapter 8](https://bit.ly/3Vp2xUO)
- [Cooking Conversions in Chapter 10](https://www.dropbox.com/s/hrqmyh62tszjqyk/conversions.json?dl=0)
- [Cooking Blog Corpus in Chapter 10](https://www.dropbox.com/sh/438c4j9lmogjcl5/AAAad7MOhkeoDYrNey3DskOoa?dl=0)


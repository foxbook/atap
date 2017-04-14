import tweepy
from slugify import slugify

API_KEY = ""
API_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

users = ["tonyojeda3","bbengfort","RebeccaBilbro","OReillyMedia",
         "datacommunitydc","dataelixir","pythonweekly","KirkDBorne"]

for user in users:
    user_timeline = api.user_timeline(screen_name=user, count=100)
    for idx, tweet in enumerate(user_timeline):
        tweet_text = user_timeline[idx].text
        filename = slugify(user + " " + tweet_text[:50]) + ".txt"
        with open(filename, 'w+') as f:
            f.write(tweet_text)

import tweepy

API_KEY             = " "
API_SECRET          = " "
ACCESS_TOKEN        = " "
ACCESS_TOKEN_SECRET = " "

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

users = ["tonyojeda3","bbengfort","RebeccaBilbro","OReillyMedia",
         "datacommunitydc","dataelixir","pythonweekly","KirkDBorne"]

def get_tweets(user_list, tweets=20):
    for user in users:
        user_timeline = api.user_timeline(screen_name=user, count=tweets)
        filename = str(user) + ".json"
        with open(filename, 'w+') as f:
            for idx, tweet in enumerate(user_timeline):
                tweet_text = user_timeline[idx].text
                f.write(tweet_text + "\n")

get_tweets(users, 100)
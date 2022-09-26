import openai
import streamlit as st
import json
import time
import plotly.graph_objs as go
import tweepy
import textblob
import pandas as pd
import re
from streamlit_lottie import st_lottie_spinner
# from keys import TWITTER_API_KEY, TWITTER_API_KEY_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, OPENAI_API_KEY

page_bg_img = """
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def summarise(text_input):
    openai.api_key = OPENAI_API_KEY

    text_input = text_input

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Summarize this in dot points:\n\n{text_input}",
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']


def conclude(text_input):
    # openai.api_key = OPENAI_API_KEY
    openai.api_key = st.secrets["OPENAI_API_KEY"]


    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Summarize the general feeling based on these tweets regard {vibe_input}:\n\n{text_input}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# api_key = TWITTER_API_KEY
# api_key_secret = TWITTER_API_KEY_SECRET
# access_token = TWITTER_ACCESS_TOKEN
# access_token_secret = TWITTER_ACCESS_TOKEN_SECRET

api_key = st.secrets["TWITTER_API_KEY"]
api_key_secret = st.secrets["TWITTER_API_KEY_SECRET"]
access_token = st.secrets["TWITTER_ACCESS_TOKEN"]
access_token_secret = st.secrets["TWITTER_ACCESS_TOKEN_SECRET"]

authenticator = tweepy.OAuthHandler(api_key, api_key_secret)
authenticator.set_access_token(access_token, access_token_secret)

api = tweepy.API(authenticator, wait_on_rate_limit=False)

st.header("WHAT'S THE VIBE???")

vibe_input = st.text_input('Please enter ANYTHING, see what twitter thinks about it:')

lottie_progress = load_lottiefile("happysad.json")

if vibe_input:
    with st_lottie_spinner(lottie_progress, loop=True, key="progress", height=250, width=250):
        time.sleep(1)
        search = f'#{vibe_input} -filter:retweets'

        tweet_cursor = tweepy.Cursor(api.search_tweets, q=search, lang='en', tweet_mode='extended').items(50)

        tweets = [tweet.full_text for tweet in tweet_cursor]

        tweets_df = pd.DataFrame(tweets, columns=['Tweets'])
        tweets_str = ''
        for _, row in tweets_df.iterrows():
            row['Tweets'] = re.sub('http\S+', '', row['Tweets'])
            row['Tweets'] = re.sub('#\S+', '', row['Tweets'])
            row['Tweets'] = re.sub('@\S+', '', row['Tweets'])
            row['Tweets'] = re.sub('\\n+', '', row['Tweets'])
            tweet = row['Tweets']
            tweets_str += f"\n{tweet}"

        sumtweet = conclude(tweets_str)
        st.write(sumtweet)

        tweets_df['Polarity'] = tweets_df['Tweets'].map(lambda tweet: textblob.TextBlob(tweet).sentiment.polarity)
        tweets_df['Result'] = tweets_df['Polarity'].map(lambda pol: '+' if pol > 0 else '-')

        positive = tweets_df[tweets_df.Result == '+'].count()['Tweets']
        negative = tweets_df[tweets_df.Result == '-'].count()['Tweets']

        hist_data = [positive, negative]
        group_labels = ['Positive', 'Negative']

        trace1 = go.Bar(
            x=["Positive words/vibes 😇"],
            y=[positive],
            name="Positive words/vibes"
        )

        trace2 = go.Bar(
            x=["Negative words/vibes 🤬"],
            y=[negative],
            name="Negative words/vibes"
        )

        data = [trace1, trace2]
        layout = go.Layout(
            barmode='group'
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="Sentiment Analysis based on Tweets",
            yaxis_title="Number of Tweets",
        )
        plot = st.plotly_chart(fig, use_container_width=True)
        st.warning("Please note that Sentiment analysis scores take into account rude and slang words as negative and may not reflect the true nature or meaning of the tweets. The summary above analyses the tweets and understands context, jokes, and slang (even rude words)!")

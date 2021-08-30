import sys
import requests
import os
import json
import ray
import time

from transformers import pipeline
from fastapi import FastAPI
from ray import serve



# ray.client("anyscale://tfu-tutorial").cluster_env("tfu-tutorial-cluster-env").namespace("summarizer").job_name("tfu-test").allow_public_internet_traffic(True).connect()
# ray.client("anyscale://webinar2?&update=True").cluster_env("tfu-tutorial-cluster-env").namespace("summarizer").allow_public_internet_traffic(True).connect()
ray.client("anyscale://webinar2").namespace("summarizer").connect()

# ray.init(
# 	address="anyscale://tfu-tutorial",
# 	namespace="summarizer",
# 	job_name="tfu-tutorial",
# 	cluster_env="tfu-tutorial-cluster-env",
# 	allow_public_internet_traffic=True
# )

serve.start(detached=True)

app = FastAPI()

def bearer_oauth(r):
	bearer_token = os.environ.get("BEARER_TOKEN")
	print("BEARER_TOKEN: " + str(bearer_token))
	r.headers["Authorization"] = f"Bearer {bearer_token}"
	return r

def fetch_tweet_text(url):
	# get tweet id
	split_url = url.split("/")
	tweet_id = split_url[5]

	# make api url 
	api_url = "https://api.twitter.com/2/tweets/" + tweet_id
	
	# get tweet text 
	response = requests.get(api_url, auth=bearer_oauth)
	response_json = response.json()
	print("RESPONSE: " + str(response_json))
	text = response_json['data']['text']
	print("Tweet text: " + text)
	return text


@serve.deployment
def sentiment_model(text: str):
	sentiment_classifier = pipeline("sentiment-analysis")
	sentiment_dict = sentiment_classifier(text)[0]
	print("SENTIMENT: " + str(sentiment_dict))
	return sentiment_dict['label']
sentiment_model.deploy()


@serve.deployment
def translate_model(text: str):
	translator = pipeline("translation_en_to_fr")
	return translator(text)
translate_model.deploy()

@serve.deployment(route_prefix="/composed", num_replicas=1)
@serve.ingress(app)
class ComposedModel:
	def __init__(self):
		self.translate_model = translate_model.get_handle()
		self.sentiment_model = sentiment_model.get_handle()


	@app.get("/")
	async def sentiment_and_translate(self, url: str):
		tweet_text = fetch_tweet_text(url)
		sentiment = await self.sentiment_model.remote(tweet_text)
		translated_text = await self.translate_model.remote(tweet_text)
		return f'Sentiment: {sentiment}, Translated Text: {translated_text}'
ComposedModel.deploy()

## EXAMPLE cURLs
#curl -X GET https://session-s2s2m3q1nqsklkugh555szeu.i.anyscaleuserdata-staging.com/serve/composed/\?url\=https://twitter.com/dog_feelings/status/1353887862580645888
#curl -X GET https://session-s2s2m3q1nqsklkugh555szeu.i.anyscaleuserdata-staging.com/serve/composed/\?url\=https://twitter.com/StarbucksNews/status/1430545403854790662

import requests
import os
import json
from OpenAIPractice.api_key import key

headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+key
}

endpoint='https://api.openai.com/v1/engines/davinci/completions'

params={
    "prompt":"Dumb Joke: I'm not a vegetarian because I love animals.I'm a vegetarian because I hate plants.\n###\nTwo-Sentence Joke: Parallel lines have so much in common. It's a shame they'll never meet.\n###\nDumb Joke: Someone stole my mood ring. I don't know how I feel about that.\n###\nDunb Joke:",
    "temperature": 0.5,
    "max_tokens": 100,
    "top_p": 1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop": ["###"]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

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
    "prompt":"Write a short story for kids about a Dog named Bingo who travels to space.\n---\n\nPage 1: Once upon a time there was a dog named Bingo.\nPage 2: He was trained by NASA to go in space.\nPage 3:",
    "temperature": 0.9,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0.7,
    "presence_penalty": 0.0,
    "stop": ["Page 11:"]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

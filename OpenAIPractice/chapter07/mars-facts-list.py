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
    "prompt":"I'm studying the planets. List things I should know about Mars\n\n1. Mars is the narest planet to Earth.\n2. Mars has seasons,dry variety (not as dump as Earth's).\n3. Mars' day is about the same length as Earth's (24.6 hours).\n4",
    "temperature": 0.0,
    "max_tokens": 100,
    "top_p": 1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop": ["11."]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

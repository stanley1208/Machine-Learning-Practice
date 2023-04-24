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
    "prompt":"Suggest a list of books that everyone should try to read in their lifetime.\n\nBooks:\n1.",
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0,
    "stop": [".\n"]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

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
    "prompt":"The following is a conversation with an AI bot. The bot is very friendly and polite.\n\nHuman: Hello, How are you?\nAI: I am doing great, thanks for asking. How can I help you today?\nHuman: I just want to talk with you.\nAI:",
    "temperature": 0.9,
    "max_tokens": 150,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.6,
    "stop": ["\n, Human:, AI:"]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

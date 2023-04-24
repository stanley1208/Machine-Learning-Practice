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
    "prompt":"Extract the name and mailing address from this email\n\n:Dear Kelly\n\n,It was great to talk to you at the seminar. I thought Jane's talk was quite good\n\n.Thank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002\n\nBest,\n\nMaya\n\nName:",
    "temperature": 0.0,
    "max_tokens": 64,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,

}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

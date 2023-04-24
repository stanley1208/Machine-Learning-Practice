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
    "prompt":"Extract the title, h1, p and body text from the following HTML document:\n\n<html><head><title>Page Title</title></head><body><h1>This is a Heading</h1><p>This is a paragraph.</p></body></html>\n\nTitle:",
    "temperature": 0.0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.0,

}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params["prompt"]+result.json()["choices"][0]["text"])

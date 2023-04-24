import requests
import json
from OpenAIPractice.api_key import key

headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+key
}

endpoint='https://api.openai.com/v1/engines/davinci/completions'



params = {
  "prompt": "Provide an ESRB rating for the following text:\n\n\"I'm going to hunt you down, and when I find you, I'll make you wish you were dead.\"\n\nESRB rating:",
  "temperature": 0.7,
  "max_tokens": 60,
  "top_p": 1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n"]
}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params['prompt'] + result.json()['choices'][0]["text"])
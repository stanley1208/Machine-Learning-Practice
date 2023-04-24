import requests
import json
from OpenAIPractice.api_key import key

headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+key
}

endpoint='https://api.openai.com/v1/engines/davinci/completions'



params = {
  "prompt": "Text:An article is a word that is used to indicate that a noun is a noun without describing it. For example, in the sentence Nick bought a dog, the article a indicates that the word dog is a noun\n\nKeywords:",
  "temperature": 0.3,
  "max_tokens": 60,
  "top_p": 1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
    "stop":[","]

}

result=requests.post(endpoint,headers=headers,data=json.dumps(params))

print(params['prompt'] + result.json()['choices'][0]["text"])
import json
import requests
import os
from OpenAIPractice.api_key import key



headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+key
}

data=json.dumps({
    "prompt":"Once upon a time",
    "max_tokens":15
})

url='https://api.openai.com/v1/engines/davinci/completions'
result=requests.post(url,headers=headers,data=data)
print(result.json())


import json
import requests
import os
from OpenAIPractice.api_key import key



headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+key
}

data=json.dumps({
    "documents":["plane","boat","spaceship","car"],
    "query":"A vehicle with wheels"
})

url='https://api.openai.com/v1/engines/davinci/search'
result=requests.post(url,headers=headers,data=data)
print(result.json())


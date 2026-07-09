import os, httpx
from dotenv import load_dotenv
load_dotenv(r'd:\data_agent_lite\backend\.env', override=True)

# Check available models
r = httpx.get('https://api.deepseek.com/models',
    headers={'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}'},
    timeout=10)
print('Status:', r.status_code)
print(r.text[:2000])

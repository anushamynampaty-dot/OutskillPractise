import os
from dotenv import load_dotenv

load_dotenv()
env_key = os.environ.get("OPENROUTER_API_KEY")
print(f"key: {env_key}")
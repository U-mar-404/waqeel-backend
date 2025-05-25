import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

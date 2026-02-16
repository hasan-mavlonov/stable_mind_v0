from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # loads .env into environment

client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1-mini",
    input="Say hi in one short sentence."
)

print(resp.output_text)
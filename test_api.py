'''
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Do you know what Yokogawa company does?"}
    ]
)

print(resp)
'''
'''
from qdrant_client import QdrantClient

# Test 1: Check disk storage
print("Test 1: Disk storage")
client_disk = QdrantClient(path="./qdrant_data")
collections = client_disk.get_collections()
print(f"Collections: {[c.name for c in collections.collections]}")

'''

'''
from qdrant_client import QdrantClient


# Test 2: Check server
print("\nTest 2: Server")
try:
    client_server = QdrantClient(url="http://localhost:6333")
    collections = client_server.get_collections()
    print(f"Collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"Server not accessible: {e}")
'''
'''
import os
from dotenv import load_dotenv

load_dotenv()

print("QDRANT_API_KEY:", os.environ.get("QDRANT_API_KEY"))
'''

import pydantic_core
print(pydantic_core.__file__)
print(pydantic_core.__version__)






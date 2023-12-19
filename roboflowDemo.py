import os

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import base64
import requests

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

ASTRA_DB_TOKEN_BASED_PASSWORD = os.environ["ASTRA_DB_TOKEN"]
ASTRA_DB_KEYSPACE = "roboflow"

SECURE_CONNECT_BUNDLE_PATH = os.environ["ASTRA_DB_SCB_LOCATION"]
ASTRA_CLIENT_ID = 'token'
ASTRA_CLIENT_SECRET = ASTRA_DB_TOKEN_BASED_PASSWORD
KEYSPACE_NAME = ASTRA_DB_KEYSPACE
TABLE_NAME = 'images'

cloud_config = {
   'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, protocol_version=4)
session = cluster.connect()

IMAGE_DIR = "images/"
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
SERVER_URL = "http://localhost:9001"

# search for cats

userInput = "cat"

while userInput != "exit":
   infer_clip_payload = {
       "text": userInput,
   }

   res = requests.post(
       f"{SERVER_URL}/clip/embed_text?api_key={API_KEY}",
       json=infer_clip_payload,
   )

   embeddings = res.json()['embeddings']

   for row in session.execute(f"SELECT name, description, item_vector FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {embeddings[0]} LIMIT 1"):
       #print("\t" + str(row))
       plt.title(row.name)
       image = mpimg.imread(IMAGE_DIR + row.name)
       plt.imshow(image)
       plt.show()

       userInput = input("Next search? ")

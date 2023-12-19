import os

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import base64
import requests

import cv2
import supervision as sv

# prepare DB

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

print(f"Creating table {TABLE_NAME} in keyspace {KEYSPACE_NAME}")
session.execute(f"CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.{TABLE_NAME} (id int PRIMARY KEY, name TEXT, description TEXT, item_vector VECTOR<FLOAT, 512>)")

print(f"Creating index image_ann_index on table {TABLE_NAME} and inserting example data")
session.execute(f"CREATE CUSTOM INDEX IF NOT EXISTS image_ann_index ON {KEYSPACE_NAME}.{TABLE_NAME}(item_vector) USING 'StorageAttachedIndex'")

print(f"Truncate table {TABLE_NAME} in keyspace {KEYSPACE_NAME}")
session.execute(f"TRUNCATE TABLE {KEYSPACE_NAME}.{TABLE_NAME}")

# load data

IMAGE_DIR = "images/"
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
SERVER_URL = "http://localhost:9001"

results = []

for i, image in enumerate(os.listdir(IMAGE_DIR)):
    #Define Request Payload
    infer_clip_payload = {
        #Images can be provided as urls or as bas64 encoded strings
        "image": {
            "type": "base64",
            "value": base64.b64encode(open(IMAGE_DIR + image, "rb").read()).decode("utf-8"),
        },
    }

    res = requests.post(
        f"{SERVER_URL}/clip/embed_image?api_key={API_KEY}",
        json=infer_clip_payload,
    )

    embeddings = res.json()['embeddings']

    image = (i, image, "description", embeddings[0])

    results.append(image)

counter = 0

for result in results:
    session.execute(f"INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (id, name, description, item_vector) VALUES {result}")
    counter = counter +1

print(f"{counter} rows loaded.\n")

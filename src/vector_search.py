import os
import base64
import oci
import shutil
import time
from functools import wraps
import numpy as np
from typing import Any, Callable, TypeVar
import oracledb
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

# Oracle Database
UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")
TABLE_NAME = os.environ["TABLE_NAME"]

# Chat AzureOpenAI
AZURE_OPENAI_ENDPOINT=os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY=os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION=os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]

# Other
IMAGE_DIRECTORY_PATH = os.environ["IMAGE_DIRECTORY_PATH"]
OCI_CONFIG_FILE = os.environ["OCI_CONFIG_FILE"]
BUCKET_NAMESPACE = os.environ["BUCKET_NAMESPACE"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
F = TypeVar('F', bound=Callable[..., Any])

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-small",
    api_version=AZURE_OPENAI_API_VERSION,
)

if __name__ == "__main__":
    try:
        with oracledb.connect(user = UN, password = PW, dsn = DSN) as connection:
            print ("Database version:", connection.version)
            vectors = OracleVS(
                client=connection,
                embedding_function=embeddings_model,
                table_name=TABLE_NAME,
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
                )
            res = vectors.similarity_search(query='test', k=3)
            print(res)
            print("Finish inserting data to database")
    except Exception as e:
        print("Error inserting data to database", e)

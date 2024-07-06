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

# OCI Object Storage
config = oci.config.from_file(file_location=OCI_CONFIG_FILE)
namespace = BUCKET_NAMESPACE
bucket_name = BUCKET_NAME
object_storage_client = oci.object_storage.ObjectStorageClient(config)
object_list = object_storage_client.list_objects(BUCKET_NAMESPACE, BUCKET_NAME).data.objects

# Azure OpenAI
chat_model = AzureChatOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-small",
    api_version=AZURE_OPENAI_API_VERSION,
)

def timer(func: F) -> None:
    """Any functions wrapper for calculate execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__} took a {elapsed_time}s")
        return result
    return wrapper

def delete_files(directory: str, file_type: str) -> None:
    """ Delete files """
    files = os.listdir(directory)
    try:
        for file in files:
            if file.endswith(file_type):
                os.remove(os.path.join(directory, file))
    except FileNotFoundError:
        print("no target files")
    except Exception as e:
        print("Error delete files", e)

def dir_check(directory: str, file_type: str) -> None:
    """ Check directory """
    if os.path.exists(directory):
        try:
            delete_files(directory, file_type)
            print(f"success delete files in {directory}")
        except FileNotFoundError:
            print("no target files")
        except Exception as e:
            print("Error delete files", e)
    else:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print("Error make dirctory ", e)

def delete_dir(directory: str) -> None:
    """ Delete directory """
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print("success delete directory")
        except Exception as e:
            print("Error delete directory", e)

@timer
def download_file(object_list: np.ndarray, directory: str, file_type: str, namespace: str, bucket_name: str) -> None:
    try:
        for object in object_list:
            if object.name.endswith(file_type):
                object_client = object_storage_client.get_object(namespace, bucket_name, object.name)
                download_file_path = os.path.join(directory, os.path.basename(object.name))
                print(f"Downloading {object.name} to {download_file_path}")
                
                with open(download_file_path, "wb") as download_file:
                    download_file.write(object_client.data.content)
    except Exception as e:
        print("Error downloading mp4", e)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@timer
def image_to_text(directory: str, image_file: str):
    image_path = f"{directory}/{image_file}"
    base64_image = encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image:"},
                {"type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
    )
    response = chat_model.invoke([message])
    print(response.content)
    return response.content

@timer
def insert_to_database(directory: str, file_type: str):
    try:
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(file_type)]
        documents = [Document(page_content=image_to_text(directory=directory, image_file=file), metadata={"image": file}) for file in image_files]

        with oracledb.connect(user = UN, password = PW, dsn = DSN) as connection:
            print ("Database version:", connection.version)
            vectors = OracleVS.from_documents(client=connection,
                                            table_name=TABLE_NAME,
                                            embedding=embeddings_model,
                                            documents=documents,
                                            distance_strategy=DistanceStrategy.DOT_PRODUCT,
                                            )
            oraclevs.create_index(connection, vectors, params={"idx_name": "hnsw_idx", "idx_type": "HNSW"})
            print("Finish inserting data to database")
    except Exception as e:
        print("Error inserting data to database", e)

if __name__ == "__main__":
    dir_check(IMAGE_DIRECTORY_PATH, ".jpg")
    download_file(object_list=object_list, directory=IMAGE_DIRECTORY_PATH, file_type=".jpg", namespace=BUCKET_NAMESPACE, bucket_name=BUCKET_NAME)
    insert_to_database(directory=IMAGE_DIRECTORY_PATH, file_type=".jpg")
    # delete_dir(IMAGE_DIRECTORY_PATH)


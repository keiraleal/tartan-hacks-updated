import csv
import os
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

def semantic_cluster_name_helper(features):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": f'''The following features are part of a semantic cluster.
            What few-word, consise semantic cluster name describes the category? Output only the highest
            likelihood name and nothing else. No explaination or chit-chat needed. \n\n {features}'''}
        ]
    )
    return response.choices[0].message.content

def semantic_cluster_name(ovlp):
    for cluster in ovlp["failure"]:
        cluster["name"] = semantic_cluster_name_helper(cluster["features"])
    for cluster in ovlp["success"]:
        cluster["name"] = semantic_cluster_name_helper(cluster["features"])
    return ovlp
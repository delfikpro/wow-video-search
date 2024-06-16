


import llm
import speech_to_text
import inference
import emb
import chromadb
import uuid
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings

chroma_client = chromadb.Client()
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embs = []
        for text in input:
            embs.append(emb.get_embeddings(text))
        return embs
collection = chroma_client.create_collection(name="embs", embedding_function=MyEmbeddingFunction())

def add_video(video_url, video_path):

    transcript = speech_to_text.extract_text(video_path)
    print("transcript ok")
    description = inference.get_description(video_path)


    print("description ok")

    relevant_queries = llm.get_relevant_queries(description, transcript, "", "")
    print("relev. queries ok")
    print(relevant_queries)

    relevant_queries.append(description[0])
    print("embs ok")

    for query in relevant_queries:
        collection.add(
            documents=[
                query,
            ],
            ids=[str(uuid.uuid4())],
            metadatas=[{"video": video_url}]
        )
    print("ok")

# while True:
#     command = input("command: ")

#     if command[0] == "+":
#         video_path = command[1:]
#         add_video(video_path)
#     if command[0] == ':':
#         print(collection.query(query_texts=command[1:], n_results=10))




from flask import Flask, request, jsonify

app = Flask("wow")

def download_file(url, filename):
    # Send a HTTP request to the URL
    response = requests.get(url, stream=True)

    # Raise an error for bad status codes
    response.raise_for_status()

    # Open the destination file in write-binary mode
    with open(filename, 'wb') as file:
        # Write the response content to the file in chunks
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded successfully and saved as {filename}")

# Example usage
download_file("https://example.com/file.zip", "file.zip")
@app.route('/add_video_url', methods=['POST'])
def add_video_url():
    video_url = request.args.get('video_url')
    download_file(video_url, "tmp.mp4")
    add_video(video_url, "tmp.mp4")
    


@app.route('/get_video_urls', methods=['GET'])
def get_video_urls():
    query = request.args.get('query')
    return collection.query(query_texts=query, n_results=10)

if __name__ == '__main__':
    app.run(debug=True)
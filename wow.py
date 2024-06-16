


import llm
import speech_to_text
import inference
import emb
import chromadb
import uuid
from chromadb import Documents, EmbeddingFunction, Embeddings

chroma_client = chromadb.Client()
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embs = []
        for text in input:
            embs.append(emb.get_embeddings(text))
        return embs
collection = chroma_client.create_collection(name="embs", embedding_function=MyEmbeddingFunction())

while True:
    command = input("command: ")

    if command[0] == "+":
        video_path = command[1:]

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
                metadatas=[{"video": video_path}]
            )
        print("ok")
    if command[0] == ':':
        print(collection.query(query_texts=command[1:], n_results=10))








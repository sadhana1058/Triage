from qdrant_client import QdrantClient

client = QdrantClient(host='localhost', port=6333)

collections = client.get_collections()
print(collections)

   # Retrieve collection info
# collection_info = client.get_collection(collection_name='support_corpus')
# print(collection_info)

#    # Retrieve vectors
# vectors = client.scroll(collection_name='support_corpus', limit=10)
# for vector in vectors:
#     print(vector.id, vector.payload)
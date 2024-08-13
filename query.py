from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, SimpleKeywordTableIndex, VectorStoreIndex, QueryBundle, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List

def load_documents(file_path: str):
    try:
        return SimpleDirectoryReader(input_dir=file_path).load_data()
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {str(e)}")

def initialize_storage_context(nodes):
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    return storage_context

def create_indices(nodes, storage_context):
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    return vector_index, keyword_index

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever: VectorIndexRetriever, keyword_retriever: KeywordTableSimpleRetriever, mode: str = "OR") -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

def create_query_engines(vector_index, keyword_index):
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
    
    response_synthesizer = get_response_synthesizer()
    
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer,
    )

    keyword_query_engine = RetrieverQueryEngine(
        retriever=keyword_retriever,
        response_synthesizer=response_synthesizer,
    )
    
    return vector_query_engine, keyword_query_engine, custom_query_engine

def main():
    documents = load_documents("resume")
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    storage_context = initialize_storage_context(nodes)
    vector_index, keyword_index = create_indices(nodes, storage_context)
    vector_query_engine, keyword_query_engine, custom_query_engine = create_query_engines(vector_index, keyword_index)

    # Example
    response = custom_query_engine.query("What is john smith's first job?")
    print(response)

if __name__ == "__main__":
    main()

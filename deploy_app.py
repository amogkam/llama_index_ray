from llama_index import (
    load_index_from_storage, 
    ServiceContext, 
    StorageContext, 
    LangchainEmbedding,
)
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from starlette.requests import Request

from ray import serve

import os
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please add the OPENAI_API_KEY environment variable to run this script. Run the following in your terminal `export OPENAI_API_KEY=...`")

openai_api_key = os.environ["OPENAI_API_KEY"]

@serve.deployment
class QADeployment:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # Define the embedding model used to embed the query.
        query_embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        service_context = ServiceContext.from_defaults(embed_model=query_embed_model)

        # Load the vector stores that were created earlier.
        storage_context = StorageContext.from_defaults(persist_dir="/tmp/ray_docs_index")
        ray_docs_index = load_index_from_storage(storage_context, service_context=service_context)   

        storage_context = StorageContext.from_defaults(persist_dir="/tmp/ray_blogs_index")
        ray_blogs_index = load_index_from_storage(storage_context, service_context=service_context)  

        # Define 2 query engines:
        #   1. Ray documentation
        #   2. Anyscale blogs
        self.ray_docs_engine = ray_docs_index.as_query_engine(similarity_top_k=5, service_context=service_context)
        self.ray_blogs_engine = ray_blogs_index.as_query_engine(similarity_top_k=5, service_context=service_context)

        # Define a sub-question query engine, that can use the individual query engines as tools.
        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.ray_docs_engine,
                metadata=ToolMetadata(name="ray_docs_engine", description="Provides information about the Ray documentation")
            ),
            QueryEngineTool(
                query_engine=self.ray_blogs_engine, 
                metadata=ToolMetadata(name="ray_blogs_engine", description="Provides information about Ray blog posts")
            ),
        ]

        self.sub_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, service_context=service_context, use_async=False)

    def query(self, engine: str, query: str):
        # Route the query to the appropriate engine.
        if engine == "docs":
            return self.ray_docs_engine.query(query)
        elif engine == "blogs":
            return self.ray_blogs_engine.query(query)
        elif engine == "subquestion":
            response =  self.sub_query_engine.query(query)
            source_nodes = response.source_nodes
            source_str = ""
            for i in range(len(source_nodes)):
                node = source_nodes[i]
                source_str += f"Sub-question {i+1}:\n"
                source_str += node.node.text
                source_str += "\n\n"
            return f"Response: {str(response)} \n\n\n {source_str}\n"

    async def __call__(self, request: Request):
        engine_to_use = request.query_params["engine"]
        query = request.query_params["query"]
        return str(self.query(engine_to_use, query))
        

# Deploy the Ray Serve application.
deployment = QADeployment.bind()
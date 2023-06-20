# llama_index_ray

## Overview of Ray and LlamaIndex
An example on how to use LlamaIndex with Ray for productionizing LLM applications.

[LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) is a data framework for building LLM applications. It provides abstractions for ingesting data from various sources, data structures for storing and indexing data, and a retrieval and query interface.

[Ray] is A general-purpose, open source, distributed compute framework for scaling AI applications in native-Python.

By using Ray with LlamaIndex, we can easily build production-quality LLM applications that can scale out to large amounts of data.


## Example

In this example, we build a Q&A system from 2 datasources: the [Ray documentation](https://docs.ray.io/en/master/), and the [Ray/Anyscale blog posts](https://www.anyscale.com/blog). 

In particular, we create a [subquestion query engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html), that can handle complex queries that involve multiple datasources. For example, a query like `"Compare and contrast how the Ray docs and the Ray blogs present Ray Serve"` requires both datasources to be queried.

### Step 1: Scalable Data Indexing

The first step is to load our data sources and create our data ingestion pipeline. This involves parsing the soruce data and embedding the data using GPUs. The embeddings are then persisted in a vector store.

LlamaIndex provides the abstraction for reading and loading the data, while [Ray Datasets](https://docs.ray.io/en/master/data/data.html) is used to scale out the processing/embedding across multiple GPUs.

Run `python create_vector_index.py` to run the data indexing.

### Step 2: Deploy the Q&A application

Next, we use LlamaIndex and [Ray Serve](https://docs.ray.io/en/master/serve/index.html) to deploy our Q&A application. 

Using LlamaIndex, we can define multiple query engines to answer questions from multiple sources. The default LLM for LlamaIndex is OpenAI GPT-3.5.
1. Ray Documentation only
2. Ray blog posts only
3. Both Ray documentation and Ray blog posts

Using Ray Serve, we can deploy this app so that we can send it query requests. For production settings, Ray Serve has built-in support for load balancing & autoscaling.

`serve run deploy_app:deployment`

### Step 3: Query the application

Finally, we can query the application. We provide a simple query script: `query.py`.

The first argument is which engine to use, either `docs`, `blogs`, or `subquestion`, which map to the three engines defined in step 2. The second argument is the query we want to send.

`python query.py "subquestion" "Can you tell me more about Aviary?"`

```
Response: 
Aviary is an open source multi-LLM serving solution developed by Anyscale. It is designed to make it easier to deploy and manage large-scale machine learning models. It provides a unified API for managing models, as well as a set of tools for monitoring and debugging model performance. Aviary also supports multiple languages, including Python, Java, and Go. 


Sub-question 1:
Sub question: Does the Ray documentation mention Aviary?
Response: 
No, the Ray documentation does not mention Aviary.

Sub-question 2:
Sub question: Are there any Ray blog posts about Aviary?
Response: 
Yes, there is a Ray blog post about Aviary. It is titled "Announcing Aviary: Open Source Multi-LLM Serving Solution" and can be found at the path /home/ray/default/llama_index_ray/www.anyscale.com/blog/announcing-aviary-open-source-multi-llm-serving-solution.html.
```

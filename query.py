"""
Usage
-----

First deploy the serve application:
`serve run deploy_app:deployment`

Then, in a separate terminal, run the following script to 
execute a query.

The first argument specifies which engine to use (Ray Docs, Ray blogs, or subquestion).
The second argument specifies the query.

python query.py "subquestion" "Compare and contrast how the Ray docs and the Ray blogs present Ray Serve"
"""

import sys

import requests

engine = sys.argv[1]
query = sys.argv[2]
response = requests.get(f"http://localhost:8000/?engine={engine}&query={query}")
print(response.text)
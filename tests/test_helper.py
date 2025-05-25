import pytest
import helper

# Example unit tests for helper.py

def test_cosine_similarity():
    vec1 = [1, 0]
    vec2 = [0, 1]
    assert helper.cosine_similarity(vec1, vec2) == 0.0

def test_load_env_vars_and_clients():
    # Assuming a function exists that returns LLM clients after environment loading
    llm_clients = helper.setup_clients()
    assert llm_clients is not null
    assert hasattr(llm_clients, 'cohere_client')


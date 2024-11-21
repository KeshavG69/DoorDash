from helper import *
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


path='DoorDashArticleSummary.csv'



df=pd.read_csv(path)

with open("summary.json", "r") as file:
    loaded_documents = json.load(file)


summary_docs = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in loaded_documents
]


with open("articles.json", "r") as file:
    loaded_documents = json.load(file)

articles_docs = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in loaded_documents
]

headings=df['Sub Article Headings'].tolist()
links=df['Sub Article Links'].tolist()
articles=df['Sub Article Text'].tolist()
summary=df['Sub Article Summaries'].tolist()


pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
vectorstore = PineconeVectorStore(index=pc.Index('doordash'), embedding=embd)

summarise_query_template='''
Summarize the following chat conversation between a DoorDash Dasher and customer support, and extract a brief issue description that captures the key problem or request. The summary should be specific and actionable, so that it can be used to search for relevant articles in a knowledge base or vector database. Avoid unnecessary details, focusing only on the essential information for article retrieval.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
Only return Summarised question and nothing else.Make sure the question compasses the entire conversation and is clear and concise.
Dont write Summary: 
only return the question and nothing else
Chat conversation:
{chat_history}

'''

summarise_query_prompt=ChatPromptTemplate.from_messages([
    ("system",summarise_query_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
summarise_chain=summarise_query_prompt|llm|StrOutputParser()



text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",

            ],
            chunk_size=1000,
            chunk_overlap=300,
        )





class Groundness(BaseModel):
  """ Binary Score for groundness check on retrieved documents and the response"""
  binary_score:str=Field(
      ...,
      description='The response by LLm  are grounded in the respect to the retrieved document "yes" or "no"'
  )


ground_structured_llm=llm90.with_structured_output(Groundness)

system = """You are a grader assessing groundness of a LLM response to retrieved documents. \n
    If the response has grounded facts based on the retrived document grade it as grounded\n
    Give a binary score 'yes' or 'no' score to indicate whether the response is grounded to the retreived document."""

ground_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n LLM Response: {llm_response}"),
    ]
)

ground_chain=ground_prompt | ground_structured_llm

class Coherence(BaseModel):
  """ Binary Score for Coherence check on retrieved documents and the response"""
  binary_score:str=Field(
      ...,
      description='The response by LLm  are Coherenent in the respect to the retrieved document "yes" or "no"'
  )


structured_llm_coherent=llm90.with_structured_output(Coherence)

system = """You are a grader assessing coherence of a LLM response to retrieved documents. \n
    If the response is coherent based on the retrived document grade it as coherent\n
    Give a binary score 'yes' or 'no' score to indicate whether the response is coherent to the retreived document."""
coherent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n LLM Response: {llm_response}"),
    ]
)

coherent_chain=coherent_prompt | structured_llm_coherent

class Compliance(BaseModel):
  """ Binary Score for Compliance of the LLM Response such that it does not have any harmful lamguage"""
  binary_score:str=Field(
      ...,
      description='The response by LLm  are compliant  such that it does not have any harmful lamguage "yes" or "no"'
  )


structured_llm_compliance=llm90.with_structured_output(Compliance)


system = """You are a grader assessing Compliance of a LLM response . \n
   The response by LLm  are compliant and  does not have any harmful lamguage grade it as compliant\n
    Give a binary score 'yes' or 'no' score to indicate whether the response is comlpiant and does not have any harmful lamguage."""
compliant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "LLM Response: {llm_response}"),
    ]
)

compliant_chain=compliant_prompt | structured_llm_compliance

class Answer(BaseModel):
  """ Binary Score for if the LLM Response answers the user query"""
  binary_score:str=Field(
      ...,
      description='The response by LLm  answer the user query "yes" or "no"'
  )


structured_llm_answer=llm90.with_structured_output(Answer)

system = """You are a grader assessing Compliance of a LLM response . \n
   The response by LLm  answers the user query\n
    Give a binary score 'yes' or 'no' score to indicate whether the response answers the users query properly."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "LLM Response: {llm_response}"),
    ]
)

answer_chain=answer_prompt | structured_llm_answer

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    query: str
    generation: str
    similarity_search: str
    groundness: str
    coherence: str
    compliance: str
    answer:str
    chat_history:List

    documents: List[Document]

def generate(state):
  gen_template=""""
<Introduction>
You are a highly intelligent and context-aware assistant specializing in generating precise and helpful answers. Your task is to use the provided article as the primary source of information and the user query to craft an accurate, clear, and concise response.
</Introduction>

<Rules>
1. Read and understand the provided article to extract relevant information.
2. Focus specifically on addressing the user's query.
3. Provide a solution, explanation, or relevant details derived from the article.
4. Use simple, user-friendly language and avoid unnecessary repetition.
5. If the article doesn’t directly address the query, use the information to provide the best alternative or related guidance.
</Rules>

<Input>
- **Article:** {docs}
- ** Query:** {query}
</Input>

<Output>
[Provide a detailed and relevant answer tailored to the query based on the article.]
</Output>


"""
  query=state['query']
  context=[]
  context_without_link=[]
  chat_history=state['chat_history']
  summarised_query=summarise_chain.invoke({'chat_history':chat_history,'question':query})
  print(summarised_query)
  print(chat_history)

  relevant_documents=vectorstore.similarity_search(summarised_query,k=7)
  keys = [doc.metadata['doc_id'] for doc in relevant_documents]
  for doc in articles_docs:


    if doc.metadata['doc_id'] in keys:
      context.append(doc)
      context_without_link.append(doc.page_content)

  response = co.rerank(
    query=query,
    documents=context_without_link,
    top_n=3,
    model='rerank-english-v3.0'
)
  top_docs = [context_without_link[result.index] for result in response.results]
  top_docs_with_link=[context[result.index] for result in response.results]

  prompt=ChatPromptTemplate.from_template(gen_template)
  gen_chain={'docs':lambda x: top_docs,'query':RunnablePassthrough()}|prompt|llm_together|StrOutputParser()
  generation=gen_chain.invoke({'query':summarised_query})
  # chat_history.append(query)
  # chat_history.append(generation)
  print(summarised_query)
  return {'documents':top_docs_with_link,'question':query,'generation':generation,'chat_history':chat_history}



def sim_search(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']

  splits=text_splitter.split_documents(documents)
  overall_similarity = compute_overall_similarity(generation, splits)
  if overall_similarity>0.70:
    print('Similarity Search Passed')
    return {'documents':documents,'question':query,'generation':generation,'similarity_search':'yes'}
  else:
    print('Similarity Search Not Passed')
    return {'documents':documents,'question':query,'generation':generation,'similarity_search':'no'}






def groundness(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']
  groundness_chain=ground_chain.invoke({'document':documents,'llm_response':generation})
  if groundness_chain.binary_score=='yes':
    print('Groundness Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','similarity_search':'no'}
  else:
    print('Groundness Not Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'no','similarity_search':'no'}

def coherence(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']
  coherence_chain=coherent_chain.invoke({'document':documents,'llm_response':generation})
  if coherence_chain.binary_score=='yes':
    print('Coherence Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'yes','similarity_search':'no'}
  else:
    print('Coherence Not Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'no','similarity_search':'no'}

def compliance(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']
  compliance_chain=compliant_chain.invoke({'llm_response':generation})
  if compliance_chain.binary_score=='yes':
    print('Compliance Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'yes','compliance':'yes','similarity_search':'no'}
  else:
    print('Compliance Not Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'yes','compliance':'no','similarity_search':'no'}

def answer(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']

  answer_chain1=answer_chain.invoke({'llm_response':generation,'query':query})

  if answer_chain1.binary_score=='yes':
    print('Answer Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'yes','compliance':'yes','similarity_search':'no','answer':'yes'}
  else:
    print('Answer Not Passed')
    return {'documents':documents,'question':query,'generation':generation,'groundness':'yes','coherence':'yes','compliance':'yes','similarity_search':'no','answer':'no'}


def send_user(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']
  print('Answer Sent to User')
  return {'generation':generation}

def human_support(state):
  documents=state['documents']
  query=state['query']
  generation=state['generation']
  return {'generation':'Human Support'}


def sim_edge(state):
  if state['similarity_search']=='yes':
    print('Send User')
    return 'send_user'
  print('Groundness')
  return 'groundness'

def ground_edge(state):
  if state['groundness']=='yes':
    print('Coherence')
    return 'coherence'
  print('Human Support')
  return 'human_support'

def coherence_edge(state):
  if state['coherence']=='yes':
    print('Compliance')
    return 'compliance'
  print('Human Support')
  return 'human_support'

def compliance_edge(state):
  if state['compliance']=='yes':
    print('Answer')
    return 'answer'
  print('Human Support')
  return 'human_support'

def answer_edge(state):
  if state['answer']=='yes':
    print('Send User')
    return 'send_user'
  print('Human Support')
  return 'human_support'

 
workflow = StateGraph(GraphState)
workflow.add_node('generate',generate)
workflow.add_node('sim_search',sim_search)
workflow.add_node('groundedness',groundness)
workflow.add_node('coherences',coherence)
workflow.add_node('compliances',compliance)
workflow.add_node('answers',answer)
workflow.add_node('send_user',send_user)
workflow.add_node('human_support',human_support)
workflow.add_edge(START, 'generate')
workflow.add_edge('generate', 'sim_search')
workflow.add_conditional_edges('sim_search',sim_edge,
                               {
                                   "groundness":'groundedness',
                                   "send_user":'send_user',

                               })
workflow.add_conditional_edges('groundedness',ground_edge,
                               {
                                   "coherence":'coherences',
                                   "human_support":'human_support',
                               })
workflow.add_conditional_edges('coherences',coherence_edge,
                               {
                                   "compliance":'compliances',
                                   "human_support":'human_support',
                               })
workflow.add_conditional_edges('compliances',compliance_edge,
                               {
                                   "answer":'answers',
                                   "human_support":'human_support',
                               })
workflow.add_conditional_edges('answers',answer_edge,
                               {
                                   "send_user":'send_user',
                                   "human_support":'human_support',
                               })

workflow.add_edge("send_user", END)
app = workflow.compile()

# chat_history=[]
# while True:
#   query=input('Input Query: ')
#   if query.lower()=='quit':
#     break
#   response=app.invoke ({'query':query,'chat_history':chat_history})
#   print(response['generation'])

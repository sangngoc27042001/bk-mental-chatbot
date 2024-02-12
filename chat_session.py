#implement chatsession
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
from langchain.chains import TransformChain
import json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pickle

openai_api_key = pickle.load(open('config.pkl' , 'rb'))
openai_api_key = openai_api_key.get('OpenAIAPIkey')

# 1. LOAD RETRIEVER

hfEmbedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vectordb = FAISS.load_local("FAISS_vectordb", hfEmbedding)
retriever = vectordb.as_retriever()

# 2. DEFINE LLM
llm = ChatOpenAI(temperature=0, openai_api_key = openai_api_key)

# 3. DEFINE NAVIGATOR
def navigator(inputs):
    out = llm.invoke(f"""Consider the conversation as follows,
for the key `real_consultant`, return boolean `true` if Human say that they want agree to connect with real mental consultant, otherwise, return `false`.
for the key `address`, return boolean `true` if Human has provided their address, otherwise, return `false`.
Desired format:
```
{{
  "real_consultant": "<real_consultant>",
  "address": "<address>"
}}
```

conversation:
```
{inputs["chat_history"]}\nHuman Message:{inputs["question"]}
```""")
    want_to_meet_real_consultant = json.loads(out.content) #parse the output in json format
    print(want_to_meet_real_consultant)
    if want_to_meet_real_consultant['real_consultant'] ==True and want_to_meet_real_consultant['address']==True:
        #TODO: Integrate google api to search for nearest real consultant here
        who = 'Vo Ngoc Sang'
        where = 'Nha Trang'
        contexts = f"Recommend {who} at {where}."
        pass
    else:
        #RAG with examples of question-answer
        #TODO: We can upgrade by using multi query here
        docs = vectordb.similarity_search_with_score(query=inputs["question"])

        docs = [f"{d[0].page_content}\n{d[0].metadata['response_j']}" for d in docs if d[1]<1]
        contexts = "\n---\n".join(docs) if len(docs)!=0 else "no context"

    docs_dict = {
        "query": inputs["question"],
        "contexts": contexts,
        "chat_history": inputs["chat_history"]
    }
    return docs_dict

navigator_chain = TransformChain(
    input_variables=["question"],
    output_variables=[],
    transform=navigator
)

# 4. DEFINE PROMPT TEMPLATE
prompt_template = PromptTemplate(
    input_variables=["query", "contexts", "chat_history"],
    template="""You are a helpful mental consultan. You will try to calm down the user as they provide their situation.
You have to introduce yourself as an AI mental consultan.
If user want to ask for real mental consultants, you have to ask the user's address first so that you could recommend near consultants.
Your answer should follow:
- provide accurate and relevant information.
- offer clear instructions and guidance.
- provide emotional support and encouragement.
- rephrase, reflect on user inputs, and exhibit active listening skills.
- analyze and interpret situations or user inputs.
- share relevant information about yourself.
- ask appropriate questions to gather necessary details.
Below is the context that contains some question-answer pairs that may relevant to the user's question. You may refer it, not copy it.
If there is no context, just behave normally.

# Contexts:
{contexts}

# Current conversation:
{chat_history}
Human Message: {query}
AI Message: """,
)

# 5. DEFINE THE FINAL CHAIN
main_chain = navigator_chain|prompt_template|llm

# 6. DEFINE THE CLASS CHAT SESSION.
#TODO: improve the ChatSession to use session id
class ChatSession:
    def __init__(self, chat_history = ""):
        self.chat_history = chat_history
        self.main_chain = navigator_chain|prompt_template|llm
    def invoke(self, user_query):
        output = self.main_chain.invoke({
            "question":user_query,
            "chat_history":self.chat_history
        })
        AIMessage = output.content
        self.chat_history += f'Human Message: {user_query}\nAI Message: {AIMessage}\n'
        return AIMessage
    def clear(self):
        self.chat_history = ""
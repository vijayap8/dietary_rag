# -*- coding: utf-8 -*-

'''Requires
langchain
langchain-core
langchain-chroma
langchain-community
langchainhub
langchain_experimental
langchain-together
'''

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_together.embeddings import TogetherEmbeddings


hf = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5")

# Knowledge base will contain documents, separated into topics by '===', with associated links or citations as headings, specified w/ '+++'
knowledge_base = '<PATH_TO_KNOWLEDGE_BASE>'
prompt_string ="You are an expert dietary assistant that gives lifestyle recommendations based on questions. Use the following pieces of retrieved context to inform your answer. If you can't figure out the answer from the context, just say 'Please reach out to a dietition or medical practitioner for answers to this question'. Cite your sources specifically by including a section called 'Citations'\nQuestion: {question} \nContext: {context} \nContect Source: {source} \nAnswer:"

# guideline fetching
with open (knowledge_base, 'r') as f:
    docs = f.read()
    docs = docs.split('===')
listofguidelines = []
for i in docs:
    guideline_link, guideline_doc = (i.split('+++'))
    current_guideline = Document(page_content=guideline_doc, metadata={"source": guideline_link})
    listofguidelines.append(current_guideline)
print("%d Guideline Sources:" % len(listofguidelines))
for i in listofguidelines:
    print(i.metadata)

fulldocs = []
fulldocs.extend(listofguidelines)

vectorstore = Chroma.from_documents(documents=fulldocs,embedding=hf)

prompt = PromptTemplate.from_template(template=prompt_string)
print(prompt)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def extract_source(docs):
    return "\n\n".join(doc.metadata['source'] for doc in docs)


prompts = ["What kinds of oils should I cook with for a heart healthy diet?",
"How can I decrease sodium in a heart healthy diet?",
"How much sodium should I eat per day on a heart healthy diet?",
"Is intermittent fasting a good way to lose weight and healthy for the heart?",
"Is the keto diet heart healthy?",
"What should I eat to lower my blood pressure?",
"What kinds of carbohydrates should I be eating in a heart healthy diet?",
"How many eggs can I eat on a heart healthy diet?",
"How much red meat should I eat for a heart healthy diet?",
"What are some good snack options that are higher in protein?",
"What are some good plant based protein sources?",
"How many meals a day should I eat for a heart healthy diet?",
"What counts as a serving of fruits and vegetables?",
"Can I eat pizza on a heart healthy diet?",
"How do I incorporate canned and frozen fruits and vegetables into my daily servings?",
"Which foods should I avoid on a heart healthy diet?",
"What diet should I follow to lower cholesterol?",
"Can I have dairy on a heart healthy diet?",
"What are some heart healthy breakfast ideas?",
"How much added sugar is okay to consume in a heart healthy diet?",
"What are healthy fats to incorporate in a heart healthy diet",
"How many calories should I be eating per day to support a heart healthy diet?",
"Which diet is best to prevent heart disease?",
"Can I drink alcohol on a heart healthy diet?",
"Should I follow a low carb diet to lower cholesterol?",
"How much protein should I eat per day as a part of a heart healthy diet?",
"Can I still drink caffeine on a heart healthy diet?",
"Can I still drink soda on a heart healthy diet?",
"Can I eat nuts on a heart healthy diet?",
"How much fish should I be eating on a heart healthy diet?",
"How can I manage my blood sugars?"]


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="{YOUR_API_KEY}",
    model="meta-llama/Llama-3-70b-chat-hf", temperature=0.5)

rag_chain = ({"question": RunnablePassthrough()} | prompt | llm | StrOutputParser())


for i in prompts:
    for attempt in range(3):
        print("\n\n Attempt %d \n" % (attempt+1))
        print("%s:\n" % i)
        for chunk in rag_chain.stream(i):
            print(chunk, end='', flush=True)

    print('\n\n\n\n NEXT QUESTION \n\n\n')


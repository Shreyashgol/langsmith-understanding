import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'Sequential llm app'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

groq_api_key = os.getenv("groq_api")
model1 = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    api_key = groq_api_key,
    temperature = 0.7

)

model2 = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    api_key = groq_api_key,
    temperature = 0.5

)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'seq model',
    'tages': ['llm app','report generation', 'summarization']
}

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

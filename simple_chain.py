from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 Interesting facts about {topic}.",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

parser = StrOutputParser()

chain = prompt | model | parser

# result = chain.invoke({"topic": "Python programming language"})

# print(result)


chain.get_graph().print_ascii()

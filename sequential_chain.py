from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed 10 point(short) report about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following report in 5 points. \n {report}",
    input_variables=["report"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

chain.get_graph().print_ascii()

result = chain.invoke({"topic": "Harry Potter"})

print(result)



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm1 = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

from langchain_core.runnables import RunnableParallel

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}.",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="Generate 5 Q&A from the following notes \n {text}.",
    input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="Merge the following notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}.",
    input_variables=["notes", "quiz"]
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain

result = chain.invoke({'text': "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a large standard library and a vibrant ecosystem of third-party packages, making it popular for web development, data analysis, artificial intelligence, scientific computing, and more. Python's syntax emphasizes code readability, which allows developers to express concepts in fewer lines of code compared to other programming languages. It is widely used in various industries and has a strong community that contributes to its growth and development."})

chain.get_graph().print_ascii()

print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm1 = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)

parser1 = StrOutputParser()

class FeedbackSentiment(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="The sentiment of the feedback, either Positive or Negative.")

parser2 = PydanticOutputParser(pydantic_object=FeedbackSentiment)

# prompt1 = PromptTemplate(
#     template="Classify the sentiment of the feedback into positive or negative. \n {feedback}",
#     input_variables=["feedback"]
# )
prompt2 = PromptTemplate(
    template="Classify the sentiment of the feedback into positive or negative. \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)


# classifier_chain1 = prompt1 | model1 | parser1
classifier_chain2 = prompt2 | model1 | parser2

# result1 = classifier_chain1.invoke({'feedback': "The product is amazing and I love it!"})
# print(result1)
# The sentiment of the feedback "The product is amazing and I love it!" is positive.


# @@@@@@@ how to handle what the llm will return -> here we want only Positive or Negative -> Pydantic validation
# result2 = classifier_chain2.invoke({'feedback': "The product is amazing and I love it!"})

# print(result2)
# sentiment='Positive'

# result = classifier_chain2.invoke({'feedback': "The product is terrible and I hate it!"}).sentiment
# print(result)
# Negative


# @@@@@@@@ Now we start the conditional branching using RunnableBranch @@@@@@@@@

# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     ........,
#     default
# )


prompt_postive = PromptTemplate(
    template="Generate a positive response to the following feedback \n {feedback}",
    input_variables=["feedback"]
)
prompt_negative = PromptTemplate(
    template="Generate a negative response to the following feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt_postive | model1 | parser1),
    (lambda x: x.sentiment == 'Negative', prompt_negative | model1 | parser1),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classifier_chain2 | branch_chain

result = chain.invoke({'feedback': "The product is terrible and I hate it!"})
print(result)

chain.get_graph().print_ascii()
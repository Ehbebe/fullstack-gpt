from typing import Any, Dict
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field  # pylint: disable=E0611
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

pc = Pinecone(api_key="PINECONE_API_KEY")

app = FastAPI(
    title="ChefGPT. 한국 음식 레시피를 제공하는 최고의 서비스입니다.",
    description="ChefGPT에게 재료 이름을 알려주면, 그 재료를 사용한 다양한 한국 음식 레시피를 알려드립니다.",
    servers=[{"url": "https://mod-sk-vary-positioning.trycloudflare.com"}],
)

embeddings = OpenAIEmbeddings()

vector_store = PineconeVectorStore.from_existing_index(
    "recipes",
    embeddings,
)


class Document(BaseModel):
    page_content: str


prompt_template = """
당신은 한국 음식 전문 셰프입니다. 주어진 재료를 사용하여 맛있는 한국 음식을 만드는 방법을 한국말로 알려주세요.

재료: {ingredient}

레시피:
"""

prompt = PromptTemplate(
    input_variables=["ingredient"],
    template=prompt_template,
)

llm = ChatOpenAI(temperature=0.1)
chain = LLMChain(llm=llm, prompt=prompt)


@app.get(
    "/recipes",
    summary="레시피 목록을 반환합니다.",
    description="재료를 입력받으면, 해당 재료를 포함하는 레시피 목록을 반환합니다.",
    response_description="레시피와 조리법을 포함하는 Document 객체",
    response_model=str,
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    recipe = chain.run(ingredient=ingredient)
    return recipe


user_token_db = {"ABCDEF": "hojin"}


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>ChefGPT 로그인</title>
        </head>
        <body>
            <h1>ChefGPT에 로그인</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">ChefGPT 승인</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    return {
        "access_token": user_token_db[code],
    }

from aiogram import Bot, Dispatcher, executor,types
from aiogram.types import Message
import asyncio
import logging
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
import os




bot = Bot(token=os.environ.get('TB_TOKEN'))

dispatcher = Dispatcher(bot)

@dispatcher.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет! Я виртуальный помощник ТюмГУ, ты можешь задать свой вопрос.")

documents = pd.read_csv("departments.csv", index_col=0).dropna(ignore_index=True)
documents.columns = ["institution", "department", "url", "description"]
loader = DataFrameLoader(documents, page_content_column='description')
loaded_documents = loader.load()
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ.get('HF_TOKEN'),
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = FAISS.from_documents(loaded_documents, embeddings)
db.as_retriever()
db.save_local('faiss_index')

@dispatcher.message_handler()
async def main(request: types.Message):
    print(request)
    question = request['text']
    found_doc = db.similarity_search(question)[0].dict()
    department = found_doc['metadata']['department']
    url = found_doc['metadata']['url']
    await request.reply(f'Возможно, тебе стоит обратиться в {department}.\nПодробнее: {url}')
async def main():
    await dispatcher.start_polling(bot)

if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates=True)

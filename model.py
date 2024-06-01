from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from chat import get_chat_responses, load_and_process_chat_history, find_answer_in_chat_history

import chainlit as cl
from db import handle_db_queries

DB_FAISS_PATH = 'vectorstore//db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        #model = "TheBloke/Llama-2-7B-Chat-GGML",
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    if chain is None:
        print("Failed to initialize chain object")
    else:
        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to BotVerse. What is your query?"
        await msg.update()

        cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    if chain is None:
        print("Error: The chain is not initialized.")
    else:
        if "Chinook DB" in message.content:  # This is a simple example condition for database queries
            response = handle_db_queries(message.content)
            answer = response.get("result") if isinstance(response, dict) else response
            sources = response.get("source_documents") if isinstance(response, dict) else None
        elif "query chat history" in message.content.lower():  # Placeholder trigger
            chat_history = load_and_process_chat_history(

                "_chat.txt")
            # Extract the actual question from the message content
            question = message.content.replace("query chat history", "").strip()
            # Find an answer in the chat history
            answer = find_answer_in_chat_history(question, chat_history)
            await cl.Message(content=answer).send()

        else:
            # Use the existing chain for other types of queries
            response = await chain.acall(message.content, callbacks=[cb])
            answer = response.get("result") if isinstance(response, dict) else response
            sources = response.get("source_documents") if isinstance(response, dict) else None

            if sources:
                answer += f"\nSources:" + str(sources)
            else:
                answer += "\nNo sources found"
        await cl.Message(content=answer).send()

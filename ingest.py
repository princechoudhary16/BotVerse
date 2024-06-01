from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = '71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf'
DB_FAISS_PATH = 'vectorstore//db_faiss'

# Create vector database
def create_vector_db():
    loader = PyPDFLoader(file_path=DATA_PATH)
    # loader = DirectoryLoader(DATA_PATH,
    #                          glob='*.pdf',
    #                          loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

#, allow_dangerous_deserialization=True

if __name__ == "__main__":
    create_vector_db()


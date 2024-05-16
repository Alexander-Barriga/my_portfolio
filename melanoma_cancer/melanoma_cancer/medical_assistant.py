# import LLM 
from langchain_openai import ChatOpenAI
# guide response using template 
from langchain_core.prompts import ChatPromptTemplate
# let's work with strings in the output
from langchain_core.output_parsers import StrOutputParser
# import embedding models
from langchain_openai import OpenAIEmbeddings
# import pdf reader
from langchain_community.document_loaders import PyPDFLoader
# import vectorstore 
from langchain_community.vectorstores import FAISS
# import multiple document text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import document template for answer retrieval 
from langchain.chains.combine_documents import create_stuff_documents_chain
# retrieval chain helper
from langchain.chains import create_retrieval_chain
# Retrieval Augmented Generation

import os
from dotenv import load_dotenv # used to load env variables 
load_dotenv()

# Retrieval Augmented Generation

class chatbot():
    def __init__(self):
            # load llm
        self.llm = ChatOpenAI() 
        self.n_docs_retrieve = 10
        self.embeddings = OpenAIEmbeddings()
        
        self.__load_docs()
        self.__set_prompt()
        
    def __load_docs(self):
        docs = []
        for doc in os.listdir(os.getenv("PDF_FOLDER_PATH")):
            if doc.endswith('.pdf'):
                pdf_path = os.path.join(os.getenv("PDF_FOLDER_PATH"), doc)
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load_and_split())
        
        self.faiss_index = FAISS.from_documents(docs, self.embeddings) 
        
    def __set_prompt(self):
        # set up the chain that takes a question and the retrieved documents and generates an answer.
        self.prompt = ChatPromptTemplate.from_template("""Your role is to provide information regarding melanoma. 
        Attempt to answer the following question based only on the provided context:

            <context>
            {context}
            </context>
            
        If the context doesn't provide an answer, provide an answer from your own knowledge.
        Question: {input}""")
        

    def get_retrieval_augmented_answer(self, query):
        
        docs = self.faiss_index.similarity_search(query, k=self.n_docs_retrieve) # find docs sim to prompt

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, self.embeddings) # we have this data indexed in a vectorstore
        

        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # create retrieval 
        retriever = vector.as_retriever() 
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # get response for prompt
        self.response = retrieval_chain.invoke({"input": query})
        return self.response["answer"]

if __name__ == "__main__":

    chat = chatbot()
    
    query = 'What is melanoma?'
    print(chat.get_retrieval_augmented_answer(query))

    query = 'What is nodular melanoma?' 
    print(chat.get_retrieval_augmented_answer(query))

    query = 'What is the probability of death of early detection of melanoma?' 
    print(chat.get_retrieval_augmented_answer(query))
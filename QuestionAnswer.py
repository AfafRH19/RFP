
from transformers import StoppingCriteria, StoppingCriteriaList,AutoConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import bitsandbytes as bnb
from fpdf import FPDF
import chainlit as cl
import bitsandbytes
import transformers
import gradio as gr
import asyncio
import torch
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ID


model_id = "meta-llama/Llama-2-13b-chat-hf"

# Bits and bytes configuration for 4-bit quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Hugging Face authentication token (replace with your actual token)
hf_auth = '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'

# Load the model configuration
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)

# Enable evaluation mode for inference
model.eval()

print(f"Model loaded on {device}")

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

# Define stopping criteria
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# Create the text generation pipeline
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    #temperature=0.01,
    max_new_tokens=512,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=generate_text)
model_id2 = "meta-llama/Llama-2-7b-chat-hf"
model_config2 = transformers.AutoConfig.from_pretrained(
    model_id2,
    token=hf_auth
)
model2 = transformers.AutoModelForCausalLM.from_pretrained(
    model_id2,
    trust_remote_code=True,
    config=model_config2,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
tokenizer2 = transformers.AutoTokenizer.from_pretrained(
    model_id2,
    token=hf_auth
)
generate_Answers = transformers.pipeline(
    model=model2,
    tokenizer=tokenizer2,
    return_full_text=False,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    #temperature=0.01,
    max_new_tokens=512,
    repetition_penalty=1.2
)
llm2 = HuggingFacePipeline(pipeline=generate_Answers)


# Directory path containing the PDFs

# Load the PDF documents
pdf_loader=DirectoryLoader('uploads/',
                       glob="*.pdf",
                       loader_cls=PyPDFLoader)
documents = pdf_loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
all_splits = text_splitter.split_documents(documents)

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Store embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

template = """<<SYS>>
You are an expert in harmonic company and your primary responsibility is to answer questions about the XOS features and compliance .

Please provide concise and succinct answers. Use the following pieces of information to answer the questions .If you can't find the answer in these  pieces of information ,  just say you don't know and invite the user to contact the XOS product owner.

If you find contradictory answers, present them all and invite the user to contact the XOS product owner.
Only respond with "Not in the documentation".If the information needed to answer the question is not contained in the document. 
Answer the question using only the information from the attached document below:XOS-Specifications-v1.22.1-EdA.pdf and XOS Advanced Media Processor-v1.21.0.pdf.

Respond in short and concise yet fully formulated sentences, being precise and accurate.
use only those documents :XOS-Specifications-v1.22.1-EdA.pdf and XOS Advanced Media Processor-v1.21.0.pdf don't use other source to answers questions ,if you dont find the answer in those documents just "Not in the documentation",Do not use other sources to answer questions.



<</SYS>>"""

prompt = template + "CONTEXT:\n\n{context}\n" + "Question : {question}" + "[\INST]"
llama_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(llm=llm2, 
                                    chain_type='stuff',
                                    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                                    chain_type_kwargs={"prompt": llama_prompt},
                                    return_source_documents=True)

def chat_bot(Question):
    if Question.lower() == 'exit':
        return 'Exiting'
    elif Question == '':
        return None
    
    result = chain({'query': Question})
    answer = result['result']
    source_documents = result['source_documents']
    
    # Extract document name/source and page numbers
    sources = []
    for doc in source_documents:
        source_info = f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'Unknown')})"
        sources.append(source_info)
    
    # Join the sources into a single string
    sources_str = "\n".join(sources)
    
    return f"Answer: {answer}\n\nSources:\n{sources_str}"
 
    
    # Extract document name/source and page numbers
    
    
dir_path = "/home/innov_user/ModelQT/test/ResultRequirement/"


# Function to generate questions from the model and pass them to the chatbot
def process_pdfs_and_ask_questions():
    results = []  # Collect results in a list
  
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        
        if file_path.lower().endswith('.pdf') and os.path.isfile(file_path):
            print(f"Processing PDF: {file_path}")
            
            pdf_loader = PyPDFLoader(file_path)
            document = pdf_loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
            all_splits = text_splitter.split_documents(document)
            
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            
            vectorstore = FAISS.from_documents(all_splits, embeddings)
            
            template = """[INST] <<SYS>>Your input document contains a list of requirements for a product or solution. 
                                Your task is to rephrase each requirement as a question directed towards the provider of the product or solution. don't forget to Start every generated question with only "Question :"   and nothing else .
                                [INST] Support for HD service launch [/INST]
                                Question : does your system support HD service launch?
                                [INST]  Capability to handle future service launches and closures [/INST]
                        """
            
            prompt = template + "CONTEXT:\n\n{context}\n" + "Question : {question}" + "[\INST]"
            llama_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
            
            chain2 = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": llama_prompt}
            )
            
            question2 = "could you generate question from each requirement?"
            
            # Generator function to yield questions one by one
            def generate_questions(chain, query):
                response = chain({"query": query})
                print(f"Response: {response['result']}") 
                questions = response['result'].split('\n')
                for question in questions:
                    cleaned_question = question.strip()
                    if cleaned_question.startswith("Question :"):
                        yield cleaned_question
                    elif any(cleaned_question.startswith(f"{i}. Question :") for i in range(1, 100)):
                        yield cleaned_question.split(" ", 1)[1].strip()
                        
            # Using the generator to get questions one by one
            question_generator = generate_questions(chain2, question2)
            
            for question in question_generator:
                print(f"Generated Question: {question}")
                chatbot_response = chat_bot(question)
                print(f"Chatbot Response: {chatbot_response}")

                # Append results to the list
                results.append({
                    "question": question,
                    "response": chatbot_response
                })
    
    return results  # Return the collected results

# Process PDFs and ask questions
process_pdfs_and_ask_questions()









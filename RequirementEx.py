import bitsandbytes
import chainlit as cl
import gradio as gr
import asyncio
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList,AutoConfig
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

import pdfkit

from PyPDF2 import PdfReader 
from pdfminer.high_level import extract_text
import io
from fpdf import FPDF

from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfFileReader

from weasyprint import HTML

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import fitz


#########################################################################################################
#HuggingFaceH4/zephyr-7b-alpha
#meta-llama/Llama-2-13b-chat-hf
#microsoft/Orca-2-13b
#google/gemma-7b
#mistralai/Mixtral-8x7B-Instruct-v0.1
#hf_auth = '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'

#device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


    
    
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fpdf import FPDF

def requirement_extraction(pdf_dir_path, result_dir_path, model_id="meta-llama/Llama-2-7b-chat-hf", embeddings_model_name="BAAI/bge-base-en-v1.5"):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    hf_auth = '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'
    
    model_config = AutoConfig.from_pretrained(
        model_id,
        token=hf_auth
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth
    )
    
    stop_list = ['\nHuman:', '\n\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False
    
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        stopping_criteria=stopping_criteria,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(pipeline=generate_text)

    for file_name in os.listdir(pdf_dir_path):
        file_path = os.path.join(pdf_dir_path, file_name)
        if file_path.lower().endswith('.pdf') and os.path.isfile(file_path):
            print(f"Processing PDF: {file_path}")
            
            # Load and split the document
            pdf_loader = PyPDFLoader(file_path)
            document = pdf_loader.load()
            
            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
            all_splits = text_splitter.split_documents(document)
            
            # Check if any text was extracted
            if not all_splits:
                print(f"Aucun texte extrait du fichier {file_name}. Passage au fichier suivant.")
                continue
            
            # Initialize embeddings and vector store
            model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs)
            
            # Generate embeddings
            try:
                embeddings_list = embeddings.embed_documents([getattr(split, 'text', None) or getattr(split, 'page_content', None) for split in all_splits])
                if not embeddings_list or len(embeddings_list[0]) == 0:
                    raise ValueError("Les embeddings gs ne sont pas correctement forms.")
            except Exception as e:
                print(f"Erreur lors de la gration des embeddings : {e}")
                continue
            
            # Initialize FAISS vector store
            try:
                vectorstore = FAISS.from_documents(all_splits, embeddings)
            except Exception as e:
                print(f"Erreur lors de la cration du vecteur FAISS : {e}")
                continue
            
            # Define the prompt
            template = """[INST] <<SYS>> Your task is to extract all the requirements expressed in the input document.
            Requirements are typically indicated by keywords such as 'must,' 'shall,' 'should be,' 'can,' 'could,' 'would like,'
            'require,' 'required', 'be capable of dealing with the following:', 'Ability to', 'expect', 'should have', 'Expected',
            'please', 'Availability of', 'should provide', 'should indicate', 'Providing', 'to consider a', 'would like to consider 
            the following options', and similar phrases.
            Aim carefully to ensure that you meet all requirements of the input document. 
            Think through this carefully and ensure to give the context for each requirement, making sure that the context accurately 
            reflects the client's needs and expectations. 
           
            Please provide as many details as possible to enable accurate requirement extraction.
            if the document is for example presentation and doesn�t contain any requirements, please answer that no requirement was found and nothing else.
            
            <<\SYS>>[/INST]
            """ 
            prompt = template + "\n\nCONTEXT:\n\n{context}\n\nQuestion: {question}[\INST]"
            llama_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k':10}),
                chain_type_kwargs={"prompt":llama_prompt}
            )
            
            question = "Could you list all the extracted requirements?"           
            response = chain({"query": question})
            response_result = response['result']
            
            # Create PDF with results
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=response_result)
            
            output_file_name = f"result_{file_name[:-4]}_response.pdf"
            output_file_path = os.path.join(result_dir_path, output_file_name)
            pdf.output(output_file_path)
            print(f"Result PDF saved to {output_file_path}")




requirement_extraction(pdf_dir_path ="/home/innov_user/ModelQT/test/RFP-main2/Split/", result_dir_path ="/home/innov_user/ModelQT/test/RFP-main2/Requiement/")
    
    


                             
          
  
    
        
        
  
        
        
        
        
      
      
      
      
      


      
      
      
      
      
      
      
      
      
      
                                  



import openai
import pickle
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage, ServiceContext
import gradio as gr
import sys
import os
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
from gradio import networking
os.environ["OPENAI_API_KEY"] = 'your key'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    #print(len(documents))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    
    index.storage_context.persist(persist_dir='./storage')
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
 
# load index
    index = load_index_from_storage(storage_context)

    return index
def chatbot_first(input_text):
    
    query_engine = index.as_query_engine() 
    response = query_engine.query(input_text)
    return response.response


index = construct_index("docs")
df = pd.read_csv('docs/data.csv')

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False) 

def pem_chatbot(history):
        
        
    #is_present = any(df['questions'].str.contains(history[-1][0], case=False))
    #if is_present:
              
    response = chatbot_first(history[-1][0])
  
    history[-1][1] = str(response)
        
    #else:
        #history[-1][1] = "Sorry, I can't answer your question."
    
    return history




themes = gr.themes.Default().set(
    body_background_fill="linear-gradient(to top, rgb(251, 194, 235) 0%, rgb(166, 193, 238) 100%)",
    body_background_fill_dark= "body_background_fill",
    block_border_width="3px",
    block_shadow="*shadow_drop_lg",
 
    
)

with gr.Blocks(theme=themes) as demo:

    gr.Markdown(
        """
         <p style="text-align: center; color: #034f84; font-weight: bold; font-size: 40px;"> Pem Chatbot </p>
    
    """
    )
    chatbot =gr.Chatbot([], elem_id="pem_chatbot").style(height=550)
    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
    
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        pem_chatbot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    


demo.launch()        

import gradio as gr
import requests

def process_pdfs(input_dir):
    url = "http://localhost:8000/process_pdfs"
    data = {
        "input_dir": input_dir[0].name,
        "output_dir": r"E:\Zephyr\data_output",
        "regenerate": True
    }
    print(data)
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['html']
    return "Failed to connect to the server"

with gr.Blocks() as iface:
    gr.Markdown("""
    # PDF to Knowledge Graph Converter

    This application allows you to upload PDF documents and generates a knowledge graph from the content of the PDFs. 
    Simply upload your PDF files and click on "Process PDFs" to get started.
    """)

    with gr.Row():
        document = gr.Files(height=100, file_types=["pdf"], interactive=True, label="Upload your PDF documents (single)")
        
    with gr.Row():
        process_btn = gr.Button("Process PDFs")
        
    html_output = gr.HTML(label="Graph Visualization")
    
    process_btn.click(
        process_pdfs,
        inputs=[document],
        outputs=html_output
    )

iface.launch()

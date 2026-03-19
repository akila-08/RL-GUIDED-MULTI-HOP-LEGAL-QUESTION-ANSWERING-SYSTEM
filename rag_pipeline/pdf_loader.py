import fitz  # PyMuPDF

def load_pdf(path):
    text = ""
    
    doc = fitz.open(path)
    
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text += page_text + "\n"
    
    return text
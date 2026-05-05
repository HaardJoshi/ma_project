import sys

try:
    import PyPDF2
    def extract_text(pdf_path):
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
except ImportError:
    try:
        import fitz # PyMuPDF
        def extract_text(pdf_path):
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
    except ImportError:
        def extract_text(pdf_path):
            import subprocess
            # Try using pdftotext or textutil (macOS native)
            try:
                res = subprocess.run(["textutil", "-convert", "txt", "-stdout", pdf_path], capture_output=True, text=True, check=True)
                return res.stdout
            except Exception as e:
                return f"Failed to extract text. Please install PyPDF2 or PyMuPDF. Error: {e}"

if __name__ == "__main__":
    pdf_path = "report/CN6000 Handbook 2025-26.pdf"
    txt_path = "report/handbook.txt"
    text = extract_text(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Extracted text to {txt_path}")

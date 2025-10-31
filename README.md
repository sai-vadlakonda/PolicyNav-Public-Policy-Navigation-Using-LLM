# PDF Insight Voice

PDF Insight Voice is a modern web application that lets you upload any PDF, ask questions about its content (by typing or speaking), and get answers both as text and as spoken audio. You can also download the answers as a PDF.

## Features
- Upload any PDF and ask questions about its content
- Ask questions by typing or using your voice (speech-to-text)
- Get answers as text and listen to them (text-to-speech)
- Download answers and sources as a PDF
- Modern, animated UI with SVG icons

## Technologies Used
- **Frontend:** HTML, CSS, JavaScript (Web Speech API, jsPDF, SVG icons)
- **Backend:** FastAPI (Python)
- **AI/ML:** LangChain, HuggingFace Embeddings, FAISS, TinyLlama (via LM Studio)
- **Other:** Jinja2 Templates, AJAX, SVG

## How It Works
1. **Upload PDF:**
   - Click the Upload button and select a PDF. The backend processes and indexes the document for fast semantic search.
2. **Ask Questions:**
   - Type your question or use the mic button for voice input. Submit to get an answer.
3. **Get Answers:**
   - Answers are generated using a language model and relevant PDF content. Listen to answers or download them as PDF.

## Setup & Run
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Start the backend server:**
   ```sh
   uvicorn main:app --reload
   ```
5. **Open the app:**
   - Go to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Notes
- For speech-to-text and text-to-speech, use Chrome or Edge for best results.
- Make sure LM Studio is running locally for TinyLlama model inference.
- All processing is local; no data is sent to third-party servers.


## License
MIT

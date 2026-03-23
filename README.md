# LSTM_NEXT_WORD_PRED
A Streamlit web application that trains custom PyTorch LSTM models directly from uploaded PDFs. It features real-time text extraction, adjustable training hyperparameters, and an interactive dashboard to generate new, stylized text using your custom-trained neural network.


# PDF-to-Text LSTM Generator 📄🧠

A Streamlit web application that trains custom PyTorch LSTM models directly from uploaded PDFs. It features real-time text extraction, adjustable training hyperparameters, and an interactive dashboard to generate new, stylized text using your custom-trained neural network.

## ✨ Features
* **Drag-and-Drop Interface:** Easily upload any PDF document directly in the browser.
* **In-Browser Training:** Trains a lightweight PyTorch LSTM model on the extracted text.
* **Custom Hyperparameters:** Adjust epochs, sequence length, max vocabulary, and token limits via the sidebar.
* **Interactive Generation:** Generate new text based on the document's vocabulary with adjustable "temperature" for creative control.

## 🛠️ Prerequisites & Setup

You will need **Python 3.8+** installed.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-nam
```

2. Create and activate a virtual environment (Recommended)

 
# On Windows:
```python -m venv venv
venv\Scripts\activate
```

# On macOS/Linux:
```python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
   ```
   streamlit
   torch
   numpy
   PyPDF2
   ```
Then run:
```pip install -r requirements.txt```


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import PyPDF2

# --- Model Definition ---
class LSTMMODEL(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMMODEL, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out


def read_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=0).detach().numpy()
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

def generate_text(model, seed_text, next_words, temperature, sequence_length, word_to_idx, idx_to_word, vocab_size):
    model.eval()
    words = seed_text.lower().split()

    for _ in range(next_words):
        seq_words = [w for w in words[-sequence_length:] if w in word_to_idx]
        
        if len(seq_words) < 1:
            next_idx = np.random.randint(0, vocab_size)
            words.append(idx_to_word[next_idx])
        else:
            def encode(seq):
                return [word_to_idx[w] for w in seq]
            
            encoded = torch.tensor([encode(seq_words)])
            with torch.no_grad():
                output = model(encoded)[0]
            next_idx = sample_with_temperature(output, temperature)
            words.append(idx_to_word[next_idx])

    return " ".join(words)

#Web App UI 
st.set_page_config(page_title="PDF LSTM Generator", layout="wide")
st.title("Train an LSTM model on Your PDF")
st.write("Upload a document, train a lightweight LSTM model in your browser, and generate new text in the style of your document.")

#1 Sidebar for Hyperparameters
st.sidebar.header("Training Parameters")
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
max_vocab = st.sidebar.number_input("Max Vocabulary Size", 100, 5000, 1000)
max_tokens = st.sidebar.number_input("Max Tokens to Process", 500, 20000, 5000)
seq_length = st.sidebar.slider("Sequence Length", 2, 10, 3)

#2 File Uploader
uploaded_file = st.file_uploader("Drag and drop a PDF file here", type="pdf")

if uploaded_file is not None:
    #process text and train
    if st.button("Extract Text & Train Model"):
        with st.spinner("Reading PDF..."):
            text = read_pdf(uploaded_file)
        
        if not text.strip():
            st.error("Could not extract text from this PDF.")
        else:
            with st.spinner("Tokenizing and building vocabulary..."):
                tokens = text.lower().replace('\n', ' ').split()
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]

                word_counts = Counter(tokens)
                vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:max_vocab]
                word_to_idx = {word: i for i, word in enumerate(vocab)}
                idx_to_word = {i: word for word, i in word_to_idx.items()}
                vocab_size = len(vocab)
                
                # Generate sequences
                sequences = []
                for i in range(len(tokens) - seq_length):
                    seq = tokens[i:i+seq_length]
                    target = tokens[i+seq_length]
                    if all(w in word_to_idx for w in seq) and target in word_to_idx:
                        sequences.append((seq, target))

            if not sequences:
                st.error("Error: Not enough valid text to generate sequences. Try a larger PDF or smaller sequence length.")
            else:
                with st.spinner(f"Training on {len(sequences)} sequences for {epochs} epochs. This might take a moment..."):
                    X = torch.tensor([[word_to_idx[w] for w in seq] for seq, _ in sequences])
                    y = torch.tensor([word_to_idx[target] for _, target in sequences])

                    model = LSTMMODEL(vocab_size, embed_size=64, hidden_size=128)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    # Simple progress bar for the UI
                    progress_bar = st.progress(0)
                    for epoch in range(epochs):
                        model.train()
                        outputs = model(X)
                        loss = criterion(outputs, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Update progress bar
                        progress_bar.progress((epoch + 1) / epochs)

                    st.success(f"Training Complete! Final Loss: {loss.item():.4f}")
                    
                    # Save the trained model and vocab to Streamlit's session state
                    st.session_state['model'] = model
                    st.session_state['word_to_idx'] = word_to_idx
                    st.session_state['idx_to_word'] = idx_to_word
                    st.session_state['vocab_size'] = vocab_size
                    st.session_state['vocab'] = vocab

# Text Generation UI  
if 'model' in st.session_state:
    st.divider()
    st.header("Generate Text -> ")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        
        default_seed = f"{st.session_state['vocab'][0]} {st.session_state['vocab'][1]}" if st.session_state['vocab_size'] >= 2 else "the"
        seed_input = st.text_input("Seed Text", value=default_seed)
    with col2:
        gen_words = st.number_input("Number of words to generate", 10, 500, 50)
    with col3:
        temp = st.slider("Temperature (Creativity)", 0.1, 2.0, 0.7)

    if st.button("Generate"):
        generated = generate_text(
            model=st.session_state['model'],
            seed_text=seed_input,
            next_words=gen_words,
            temperature=temp,
            sequence_length=seq_length,
            word_to_idx=st.session_state['word_to_idx'],
            idx_to_word=st.session_state['idx_to_word'],
            vocab_size=st.session_state['vocab_size']
        )
        st.info(generated)

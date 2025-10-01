import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import requests
import re
import string
import os

# 1. DATASET LOADING AND PREPROCESSING

def download_shakespeare_text():
    """Download Shakespeare's complete works from Project Gutenberg"""
    # URL for plain text UTF-8 version
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    
    print("Downloading Shakespeare's works...")
    response = requests.get(url)
    
    if response.status_code == 200:
        text = response.text
        print(f"Downloaded {len(text)} characters")
        
        # Save the text to a file
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return text
    else:
        print(f"Failed to download: Status code {response.status_code}")
        return None

def preprocess_text(text):
    """Preprocess the text: lowercase, remove headers/footers, punctuation"""
    # Find the start of the actual content (after Project Gutenberg header)
    start_marker = "THE SONNETS"
    end_marker = "End of Project Gutenberg's The Complete Works of William Shakespeare"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_sequences(text, seq_length=50):
    """Create input-output pairs for training"""
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    
    # Create sequences
    input_sequences = []
    for i in range(0, len(text.split()) - seq_length):
        seq = text.split()[i:i+seq_length+1]
        input_sequences.append(seq)
    
    # Create input-output pairs
    X = []
    y = []
    for seq in input_sequences:
        X.append(seq[:-1])
        y.append(seq[-1])
    
    # Convert to numerical sequences
    X_num = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_num, maxlen=seq_length)
    
    # One-hot encode the outputs
    y_num = tokenizer.texts_to_sequences(y)
    y_one_hot = tf.keras.utils.to_categorical(y_num, num_classes=total_words)
    
    return X_pad, y_one_hot, tokenizer, total_words

# 2. MODEL DESIGN

def build_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """Build an LSTM-based text generation model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# 3. MODEL TRAINING

def train_model(model, X, y, epochs=50, batch_size=64):
    """Train the LSTM model with early stopping and checkpoints"""
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'shakespeare_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1
    )
    
    # Split data into training and validation sets
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    return history

# 4. TEXT GENERATION

def generate_text(model, tokenizer, seed_text, next_words=100, seq_length=50):
    """Generate new text based on a seed sequence"""
    # Process the seed text
    seed_text = seed_text.lower()
    seed_text = seed_text.translate(str.maketrans('', '', string.punctuation))
    
    generated_text = seed_text
    
    # Generate text word by word
    for _ in range(next_words):
        # Tokenize the current text
        token_list = tokenizer.texts_to_sequences([generated_text.split()[-seq_length:]])[0]
        
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=seq_length)
        
        # Predict the next word
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        # Convert the predicted word index to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Add the predicted word to the generated text
        generated_text += " " + output_word
    
    return generated_text

def main():
    # Check if we already have the Shakespeare text
    if not os.path.exists("shakespeare.txt"):
        text = download_shakespeare_text()
    else:
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    print(f"Processed text length: {len(processed_text)} characters")
    
    # Create sequences
    seq_length = 50
    X, y, tokenizer, total_words = create_sequences(processed_text, seq_length)
    print(f"Vocabulary size: {total_words}")
    print(f"Number of sequences: {len(X)}")
    
    # Build the model
    model = build_model(total_words, seq_length)
    model.summary()
    
    # Train the model
    history = train_model(model, X, y, epochs=30, batch_size=128)
    
    # Generate text with different seed inputs
    seed_texts = [
        "to be or not to be",
        "all the worlds a stage",
        "what light through yonder window"
    ]
    
    print("\nGenerated Text Samples:")
    for seed in seed_texts:
        generated = generate_text(model, tokenizer, seed, next_words=50, seq_length=seq_length)
        print(f"\nSeed: {seed}")
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main() 
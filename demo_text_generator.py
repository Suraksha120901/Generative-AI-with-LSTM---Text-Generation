import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import os

def preprocess_text(text, max_length=20000):
    """Preprocess a small portion of the text for demo purposes"""
    # Find the start of the actual content (after Project Gutenberg header)
    start_marker = "THE SONNETS"
    end_marker = "End of Project Gutenberg's The Complete Works of William Shakespeare"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    # Use only a portion of the text to speed up demo
    text = text[:max_length]
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_sequences(text, seq_length=10):
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

def build_small_model(vocab_size, seq_length, embedding_dim=32, lstm_units=64):
    """Build a smaller LSTM model for demo purposes"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def generate_text(model, tokenizer, seed_text, next_words=20, seq_length=10):
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
    # Load the Shakespeare text
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Preprocess a small portion of the text
    processed_text = preprocess_text(text, max_length=50000)
    print(f"Processed text length: {len(processed_text)} characters")
    
    # Create sequences
    seq_length = 10
    X, y, tokenizer, total_words = create_sequences(processed_text, seq_length)
    print(f"Vocabulary size: {total_words}")
    print(f"Number of sequences: {len(X)}")
    
    # Build the model
    model = build_small_model(total_words, seq_length)
    model.summary()
    
    # Train the model (with fewer epochs for demo)
    print("\nTraining the model (this might take a minute)...")
    model.fit(X, y, epochs=10, batch_size=128, verbose=1)
    
    # Generate text with different seed inputs
    seed_texts = [
        "to be or not to be",
        "all the worlds a stage",
        "what light through yonder window"
    ]
    
    print("\nGenerated Text Samples:")
    for seed in seed_texts:
        generated = generate_text(model, tokenizer, seed, next_words=20, seq_length=seq_length)
        print(f"\nSeed: {seed}")
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main() 
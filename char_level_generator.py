import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import time

# Import download function
from shakespeare_text_generator import download_shakespeare_text

def preprocess_text(text):
    """Simplified preprocessing for character-level model"""
    # Find the start of the actual content (after Project Gutenberg header)
    start_marker = "THE SONNETS"
    end_marker = "End of Project Gutenberg's The Complete Works of William Shakespeare"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    return text

def create_char_sequences(text, seq_length=100):
    """Create character-level sequences for training"""
    # Get unique characters
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    
    # Create sequences
    input_sequences = []
    output_chars = []
    
    for i in range(0, len(text) - seq_length):
        input_seq = text[i:i+seq_length]
        output_char = text[i+seq_length]
        
        input_sequences.append([char_to_idx[char] for char in input_seq])
        output_chars.append(char_to_idx[output_char])
    
    # Convert to numpy arrays
    X = np.array(input_sequences)
    y = tf.keras.utils.to_categorical(output_chars, num_classes=len(chars))
    
    return X, y, char_to_idx, idx_to_char, len(chars)

def build_char_model(vocab_size, seq_length, embedding_dim=64, lstm_units=256):
    """Build a character-level LSTM model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def generate_char_text(model, char_to_idx, idx_to_char, seed_text, next_chars=200, seq_length=100, temperature=1.0):
    """Generate text using character-level predictions"""
    # Ensure the seed text is exactly seq_length characters
    seed_text = seed_text[-seq_length:] if len(seed_text) > seq_length else seed_text.rjust(seq_length)
    
    generated_text = seed_text
    
    # Generate characters one by one
    for _ in range(next_chars):
        # Convert the current sequence to numerical format
        x_pred = np.array([[char_to_idx.get(char, 0) for char in generated_text[-seq_length:]]])
        
        # Predict the next character
        predictions = model.predict(x_pred, verbose=0)[0]
        
        # Apply temperature to predictions
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Sample with temperature
        next_index = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_index]
        
        # Add the predicted character to the generated text
        generated_text += next_char
    
    return generated_text[-next_chars-seq_length:]

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
    
    # Create character sequences
    seq_length = 100
    X, y, char_to_idx, idx_to_char, vocab_size = create_char_sequences(processed_text, seq_length)
    print(f"Vocabulary size: {vocab_size} characters")
    print(f"Number of sequences: {len(X)}")
    
    # Build the model
    model = build_char_model(vocab_size, seq_length)
    model.summary()
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'char_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1
    )
    
    # Split data into training and validation sets
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    
    # Load the best model
    model.load_weights('char_model.h5')
    
    # Generate text with different temperatures
    seed_texts = [
        "To be, or not to be, that is the question",
        "All the world's a stage, and all the men and women merely players",
        "What light through yonder window breaks? It is the east, and Juliet is the sun"
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    print("\nGenerated Text Samples:")
    for seed in seed_texts:
        print(f"\nSeed: {seed[:50]}...")
        
        for temp in temperatures:
            generated = generate_char_text(model, char_to_idx, idx_to_char, seed, 
                                         next_chars=200, seq_length=seq_length, temperature=temp)
            print(f"\nTemperature {temp}:")
            print(generated)

if __name__ == "__main__":
    main() 
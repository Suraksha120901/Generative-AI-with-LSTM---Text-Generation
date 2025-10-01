import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import time

# Import functions from the main script
from shakespeare_text_generator import (
    download_shakespeare_text, 
    preprocess_text, 
    create_sequences, 
    generate_text
)

def build_basic_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """Basic model with a single LSTM layer"""
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
    
    return model, "Basic_LSTM"

def build_deep_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """Deep model with two LSTM layers"""
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
    
    return model, "Deep_LSTM"

def build_bidirectional_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """Model with bidirectional LSTM layers"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Bidirectional(LSTM(lstm_units)),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model, "Bidirectional_LSTM"

def build_gru_model(vocab_size, seq_length, embedding_dim=100, gru_units=150):
    """Model using GRU instead of LSTM layers"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        GRU(gru_units, return_sequences=True),
        GRU(gru_units),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model, "GRU"

def build_dropout_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """Model with dropout for regularization"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        LSTM(lstm_units),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model, "LSTM_with_Dropout"

def train_model(model, X, y, epochs=30, batch_size=128, model_name="model"):
    """Train the model and record performance metrics"""
    # Create directory for model checkpoints if it doesn't exist
    if not os.path.exists('model_checkpoints'):
        os.makedirs('model_checkpoints')
        
    # Split data into training and validation sets
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Save the model
    model.save(f'model_checkpoints/{model_name}.h5')
    
    return history, training_time

def compare_model_performance(histories, training_times, model_names):
    """Compare and visualize model performance"""
    plt.figure(figsize=(12, 10))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        plt.plot(history.history['accuracy'], label=f'{name} Training')
        plt.plot(history.history['val_accuracy'], label=f'{name} Validation')
    
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        plt.plot(history.history['loss'], label=f'{name} Training')
        plt.plot(history.history['val_loss'], label=f'{name} Validation')
    
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Print training times
    print("Training Times:")
    for name, time_taken in zip(model_names, training_times):
        print(f"{name}: {time_taken:.2f} seconds")

def generate_samples(models, tokenizer, seed_texts, seq_length=50, model_names=None):
    """Generate text samples using different models"""
    samples = {}
    
    for i, model in enumerate(models):
        model_name = model_names[i] if model_names else f"Model {i+1}"
        samples[model_name] = []
        
        for seed in seed_texts:
            generated = generate_text(model, tokenizer, seed, next_words=50, seq_length=seq_length)
            samples[model_name].append((seed, generated))
            
    return samples

def print_samples(samples):
    """Print generated text samples"""
    for model_name, model_samples in samples.items():
        print(f"\n=== {model_name} Generated Samples ===")
        
        for seed, generated in model_samples:
            print(f"\nSeed: {seed}")
            print(f"Generated: {generated}")

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
    
    # Create sequences with different sequence lengths
    seq_lengths = [30, 50, 70]
    datasets = {}
    
    for seq_length in seq_lengths:
        X, y, tokenizer, total_words = create_sequences(processed_text, seq_length)
        datasets[seq_length] = (X, y, tokenizer, total_words)
        print(f"Sequence length {seq_length}: {len(X)} sequences created")
    
    # Select a sequence length for the model comparison
    selected_seq_length = 50
    X, y, tokenizer, total_words = datasets[selected_seq_length]
    
    # Define the models to compare
    model_builders = [
        build_basic_model,
        build_deep_model,
        build_bidirectional_model,
        build_gru_model,
        build_dropout_model
    ]
    
    # Train and evaluate each model
    histories = []
    training_times = []
    model_names = []
    trained_models = []
    
    for builder in model_builders:
        model, name = builder(total_words, selected_seq_length)
        print(f"\nTraining {name}...")
        model.summary()
        
        history, training_time = train_model(model, X, y, epochs=15, batch_size=128, model_name=name)
        
        histories.append(history)
        training_times.append(training_time)
        model_names.append(name)
        trained_models.append(model)
    
    # Compare model performance
    compare_model_performance(histories, training_times, model_names)
    
    # Generate text samples
    seed_texts = [
        "to be or not to be",
        "all the worlds a stage",
        "what light through yonder window"
    ]
    
    samples = generate_samples(trained_models, tokenizer, seed_texts, 
                              seq_length=selected_seq_length, model_names=model_names)
    print_samples(samples)
    
    # Additional experiment: Compare different sequence lengths
    print("\n\n=== Comparing Different Sequence Lengths ===")
    seq_models = []
    seq_model_names = []
    
    for seq_length in seq_lengths:
        X, y, tokenizer, total_words = datasets[seq_length]
        
        model, _ = build_deep_model(total_words, seq_length)
        model_name = f"Deep_LSTM_seq_{seq_length}"
        
        print(f"\nTraining {model_name}...")
        history, _ = train_model(model, X, y, epochs=15, batch_size=128, model_name=model_name)
        
        seq_models.append(model)
        seq_model_names.append(model_name)
    
    # Generate samples with different sequence lengths
    seq_samples = {}
    
    for i, (seq_length, model) in enumerate(zip(seq_lengths, seq_models)):
        model_name = seq_model_names[i]
        seq_samples[model_name] = []
        
        X, y, tokenizer, total_words = datasets[seq_length]
        
        for seed in seed_texts:
            generated = generate_text(model, tokenizer, seed, next_words=50, seq_length=seq_length)
            seq_samples[model_name].append((seed, generated))
    
    print_samples(seq_samples)

if __name__ == "__main__":
    main() 
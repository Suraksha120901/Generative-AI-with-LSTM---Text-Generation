# LSTM Text Generator: Task Completion Summary

This project implements an LSTM-based text generator as specified in the interview task requirements. Here's a summary of what has been accomplished:

## 1. Dataset Loading and Preprocessing ✅
- Implemented `download_shakespeare_text()` function to fetch the complete works of Shakespeare from Project Gutenberg
- Created comprehensive text preprocessing with `preprocess_text()`:
  - Lowercase conversion
  - Punctuation removal
  - Extra whitespace removal
- Built sequence generation function `create_sequences()` that:
  - Tokenizes text into words
  - Creates input-output pairs
  - Prepares one-hot encoded outputs

## 2. Model Design ✅
- Implemented LSTM-based model architecture with:
  - Embedding layer
  - One or more LSTM layers
  - Dense output layer with softmax activation
- Used categorical crossentropy loss and Adam optimizer
- Created multiple model variations with different architectures

## 3. Model Training ✅
- Implemented training functionality with:
  - Training/validation split
  - Early stopping and model checkpoints
  - Configurable batch size and epochs
- Added performance tracking and model saving

## 4. Text Generation ✅
- Implemented text generation from seed sequences
- Used iterative prediction to generate new text word-by-word
- Added temperature parameter for character-level generation to control randomness

## Bonus Tasks ✅
- Created multiple model architectures:
  - Basic single-layer LSTM
  - Deep stacked LSTM
  - Bidirectional LSTM
  - GRU-based model
  - LSTM with dropout
- Implemented character-level text generation in addition to word-level
- Added visualization of model performance metrics
- Experimented with different sequence lengths to analyze impact on quality

## Files Delivered
- `shakespeare_text_generator.py`: Main implementation 
- `char_level_generator.py`: Character-level generation
- `model_variations.py`: Experiments with different architectures
- `demo_text_generator.py`: Small demo with reduced dataset
- `requirements.txt`: Dependencies
- `README.md`: Documentation

## Dataset
The project uses Shakespeare's complete works from Project Gutenberg:
- [The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100)

This implementation fulfills all the requirements specified in the task, including the bonus experiments with different model architectures. 
# Shakespeare Text Generator using LSTM

This project implements a text generator using Long Short-Term Memory (LSTM) neural networks. The model is trained on Shakespeare's complete works and generates new text based on seed inputs.

## Dataset

The dataset used is Shakespeare's complete works from Project Gutenberg:
- [The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100)

The code automatically downloads the dataset if it's not already present.

## Project Structure

- `shakespeare_text_generator.py`: Main Python script containing word-level text generation
- `char_level_generator.py`: Character-level text generation implementation
- `model_variations.py`: Script to experiment with different model architectures (bonus task)
- `shakespeare.txt`: Downloaded Shakespeare's works (created when the script runs)
- `shakespeare_model.h5`: Trained word-level model checkpoint
- `char_model.h5`: Trained character-level model checkpoint
- `model_checkpoints/`: Directory containing saved model variations
- `model_comparison.png`: Visualization of model performance

## Requirements

The following libraries are required to run this code:
```
numpy
tensorflow
requests
matplotlib
```

You can install these dependencies with:
```bash
pip install -r requirements.txt
```

## Implementation Details

The project implements two approaches to text generation:

### Word-Level Text Generation (Main Implementation)
1. **Data Preprocessing**:
   - Downloads Shakespeare's complete works from Project Gutenberg
   - Preprocesses the text (lowercase, removes punctuation, etc.)
   - Tokenizes text into words and creates sequences

2. **Model Architecture**:
   - Embedding layer to convert word indices to dense vectors
   - Two stacked LSTM layers to capture temporal dependencies
   - Dense output layer with softmax activation for word prediction

3. **Training & Generation**:
   - Trains using categorical cross-entropy loss and Adam optimizer
   - Generates text by predicting the next word in sequence iteratively

### Character-Level Text Generation
1. **Data Preprocessing**:
   - Uses the same dataset but creates character-level sequences
   - Each character is treated as a token

2. **Model Architecture**:
   - Embedding layer for characters
   - Two LSTM layers with dropout for regularization
   - Dense output layer for character prediction

3. **Text Generation**:
   - Implements temperature parameter to control generation randomness
   - Generates text one character at a time

## Usage

To run the word-level implementation:
```bash
python shakespeare_text_generator.py
```

To run the character-level implementation:
```bash
python char_level_generator.py
```

To experiment with different model architectures:
```bash
python model_variations.py
```

## Model Variations (Bonus)

The `model_variations.py` script trains and compares five different model architectures:
1. Basic LSTM (single LSTM layer)
2. Deep LSTM (two stacked LSTM layers)
3. Bidirectional LSTM
4. GRU (Gated Recurrent Unit)
5. LSTM with Dropout

It also compares the impact of different sequence lengths (30, 50, and 70) on text generation.

The script generates:
- Performance comparison plots in `model_comparison.png`
- Sample generated texts from each model
- Saved model checkpoints in the `model_checkpoints/` directory

## Character-Level vs. Word-Level Generation

The project demonstrates both word-level and character-level text generation approaches:

**Word-Level Advantages:**
- Better at capturing semantic meaning
- Requires less training data
- Faster to train with smaller vocabulary size

**Character-Level Advantages:**
- Can generate new words not in the training vocabulary
- No out-of-vocabulary issues
- Captures spelling patterns and linguistic structure

The character-level model also implements a temperature parameter to control the randomness of generation - higher values produce more creative but potentially less coherent text.

## Customizing the Models

You can modify various parameters in the code to experiment further:
- `seq_length`: Length of input sequences
- `embedding_dim`: Dimension of word/character embeddings
- `lstm_units`: Number of LSTM units in each layer
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `temperature`: (Character-level only) Controls randomness of generation 
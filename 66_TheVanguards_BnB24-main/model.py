import os
import fitz
import re
import tensorflow as tf
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.layers import Embedding, LSTM, Dense, Dropout
from tf.keras.models import Sequential
from tf.keras.utils import to_categorical
import numpy as np

# Data Preprocessing
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def get_labels(text):
    if re.search(r'education', text, re.IGNORECASE):
        return 1
    elif re.search(r'experience', text, re.IGNORECASE):
        return 2
    elif re.search(r'skills', text, re.IGNORECASE):
        return 3
    else:
        return 0

def process_data(folder_path):
    texts = []
    labels = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            label = get_labels(text)
            if label != 0:
                texts.append(text)
                labels.append(label)
    return texts, labels

# Path to the folder containing the PDF files
pdf_folder = 'data/HR'

texts, labels = process_data(pdf_folder)

# Tokenization
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Padding sequences
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# One-hot encoding labels
labels = to_categorical(labels)

# Model architecture
embedding_dim = 100
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_sequence_length),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes (0-3)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32
model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the model
model.save('resume_classification_model.h5')

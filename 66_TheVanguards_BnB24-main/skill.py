import PyPDF2
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def extract_text_from_pdf(file_path):
    # Open the PDF file in binary mode
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the extracted text
        text = ''

        # Loop through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Return the extracted text
        return text

def grade_document(file_path, words):
    # Extract text from the PDF file
    text = extract_text_from_pdf(file_path)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Count the occurrences of the specified words
    word_counts = {word: tokens.count(word) for word in words}

    # Calculate the total count of the specified words
    total_count = sum(word_counts.values())

    # Calculate the grade based on the frequency of the specified words
    grade = total_count / len(tokens)

    return grade, word_counts

# Path to the PDF file
pdf_file_path = 'Lloyd Louis.pdf'

# Words to check for in the document
words_to_check = ['Python', 'Java', 'Html']

# Grade the document based on the frequency of the specified words
grade, word_counts = grade_document(pdf_file_path, words_to_check)

# Print the grade and word counts
print(f'Grade: {grade}')
print('Word counts:')
for word, count in word_counts.items():
    print(f'{word}: {count}')


import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import docx2txt
import re
import operator
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

background_color = (255, 255, 255)  # Assuming the background is white, but adjust this as needed

def ignore_background_color(doc):
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("dict")

        # Check the color of each text element
        for block in text["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    color = span["color"]
                    if color == background_color:
                        # Ignore this text element
                        span["flags"] |= 1 << 31  # Set the "to_ignore" flag

def read_pdf_resume(pdf_doc):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(pdf_doc, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True,check_extractable=True):           
            page_interpreter.process_page(page)     
        text = fake_file_handle.getvalue() 
    # close open handles      
    converter.close() 
    fake_file_handle.close() 
    if text:     
        return text

def read_word_resume(word_doc):
    resume = docx2txt.process(word_doc)
    resume = str(resume)
    #print(resume)
    text =  ''.join(resume)
    text = text.replace("\n", "")
    if text:
        return text
    
def clean_job_decsription(jd):
    clean_jd = jd.lower()
    clean_jd = re.sub(r'[^\w\s]', '', clean_jd)
    clean_jd = clean_jd.strip()
    clean_jd = re.sub('[0-9]+', '', clean_jd)
    clean_jd = word_tokenize(clean_jd)
    stop = stopwords.words('english')
    clean_jd = [w for w in clean_jd if not w in stop] 
    return(clean_jd)

def create_word_cloud(jd):
    corpus = jd
    fdist = FreqDist(corpus)
    words = ' '.join(corpus)
    words = words.split()
    data = dict() 
    for word in (words):     
        word = word.lower()     
        data[word] = data.get(word, 0) + 1 
    dict(sorted(data.items(), key=operator.itemgetter(1),reverse=True)) 
    word_cloud = WordCloud(width = 800, height = 800, 
    background_color ='white',max_words = 500) 
    word_cloud.generate_from_frequencies(data) 
    
    plt.figure(figsize = (10, 8), edgecolor = 'k')
    plt.imshow(word_cloud,interpolation = 'bilinear')  
    plt.axis("off")  
    plt.tight_layout(pad = 0)
    plt.show()
    
def extract_resume_sections(text):
    # Regular expressions to identify common resume sections
    section_patterns = {
        'Experience': re.compile(r'experience', re.IGNORECASE),
        'Education': re.compile(r'education', re.IGNORECASE),
        'Skills': re.compile(r'skill', re.IGNORECASE),
        'Projects': re.compile(r'project', re.IGNORECASE),
        'Certifications': re.compile(r'certification?', re.IGNORECASE),
        'Languages': re.compile(r'language', re.IGNORECASE),
        # Add more section patterns as needed
    }

    # Initialize a dictionary to store section content
    sections = {section: '' for section in section_patterns}
    # Iterate over each section pattern
    for section, pattern in section_patterns.items():
        # Find the start and end indices of the section content
        match = pattern.search(text)
        if match:
            start = match.start()
            next_match = next(pattern.finditer(text, start + 1), None)
            end = next_match.start() if next_match else None
            # Extract the section content and store it in the dictionary
            sections[section] = text[start:end].strip()
    return sections

def get_resume_score(text):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text)
    print("\nSimilarity Scores:")
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2) 
    print("Your resume matches about "+ str(matchPercentage)+ "% of the job description.")
    
def clean_text(text):
    # Remove leading and trailing whitespace
    cleaned_text = text.strip()
    # Remove extra spaces between words
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Remove newline characters
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text

def skills_match(description, skills):
    job_skills = set(description)
    
    # Split the resume skills string into individual skills
    resume_skill_list = skills.split(' ')

    # Iterate over each skill in the job description
    for job_skill in job_description:
        # Normalize the job skill by converting to lowercase and removing leading/trailing spaces
        normalized_job_skill = job_skill.lower().strip()

        # Iterate over each skill in the resume
        for resume_skill in resume_skill_list:
            # Normalize the resume skill by converting to lowercase and removing leading/trailing spaces
            normalized_resume_skill = resume_skill.lower().strip()

            # Use SequenceMatcher to calculate the similarity between the job skill and the resume skill
            similarity_ratio = SequenceMatcher(None, normalized_job_skill, normalized_resume_skill).ratio()

            # If the similarity ratio is above a threshold (e.g., 0.7), consider it a match
            if similarity_ratio > 0.7:
                matching_skills_count += 1
                break 
    
    return matching_skills_count


if __name__ == '__main__':
    extn = input("Enter File Extension: ")
    #print(extn)
    if extn == "pdf":
        resume = read_pdf_resume('Lloyd Louis.pdf')
    else:
        resume = read_word_resume('test_resume.docx')
    job_description = input("Enter the Job Description: ") 
    ## Get a Keywords Cloud 
    clean_jd = ignore_background_color(clean_job_decsription(job_description))
    text = [resume, job_description] 
    ## Get a Match score
    get_resume_score(text)
    cleaned_text = clean_text(resume)
    print()




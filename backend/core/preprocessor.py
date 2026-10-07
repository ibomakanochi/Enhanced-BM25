import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data on first run
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text):
        if not text:
            return []
            
        # 1. Lowercase and remove punctuation (keep only alphanumeric)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 2. Tokenize by whitespace
        tokens = text.split()
        
        # 3. Remove stopwords and apply Porter Stemming (The Lucene Standard)
        processed_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        return processed_tokens
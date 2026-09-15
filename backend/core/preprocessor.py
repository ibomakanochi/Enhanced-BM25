import re
import nltk
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        # Ensure dependencies are downloaded locally
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        
        # We REMOVED the PorterStemmer. 
        # FastText and NPMI lookup require whole, structurally intact words to find accurate synonyms (SOP 1).

    def clean(self, text):
        # 1. Lowercase the text
        text = text.lower()
        
        # 2. Domain-Aware Regex: Keep letters, numbers, spaces, AND hyphens (for terms like "covid-19" or "il-6")
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        
        # 3. Tokenize by splitting
        tokens = text.split()
        
        # 4. Remove stopwords and strip dangling hyphens, leaving original scientific nouns intact
        clean_tokens = [
            w.strip('-') for w in tokens 
            if w not in self.stop_words and len(w.strip('-')) > 1
        ]
        
        return clean_tokens
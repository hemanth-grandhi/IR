import re
import string

class Preprocessor:
    def __init__(self, use_stop_words=True, use_stemming=True):
        self.use_stop_words = use_stop_words
        self.use_stemming = use_stemming
        
        # A basic set of english stop words to avoid NLTK download issues at runtime if not downloaded
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", 
            "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", 
            "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", 
            "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", 
            "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", 
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", 
            "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
            "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", 
            "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", 
            "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
            "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", 
            "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", 
            "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
            "you've", "your", "yours", "yourself", "yourselves"
        }
        
        if self.use_stemming:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()

    def process(self, text):
        if not text:
            return []
            
        # 1. Case folding
        text = text.lower()
        
        # 2. Tokenization and Normalization
        # Remove punctuation by replacing with space
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
        # Split into tokens (implicitly removes extra whitespace)
        tokens = text.split()
        
        # 3. Stop-word removal
        if self.use_stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
            
        # 4. Stemming/Lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
            
        return tokens

if __name__ == "__main__":
    p = Preprocessor()
    original = "Efficient Distributed Computing for AI Models"
    tokens = p.process(original)
    print("Original:", original)
    print("Preprocessed:", " ".join(tokens))

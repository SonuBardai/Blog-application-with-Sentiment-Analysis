import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob

def clean(content):
    content = content.lower()       # Lowercase

    for letter in content:          # Symbols
        if letter in string.punctuation:
            content = content.replace(letter, "")
    
    stop_words = set(stopwords.words('english'))  # Initialize StopWords
    tokens = word_tokenize(content) # Create Tokens
    clean_tokens = list()
    for token in tokens:
        if token not in stop_words:
            clean_tokens.append(token)
    
    clean_content = " ".join(token for token in clean_tokens)

    return clean_content
    

def analysis(content):
    content = clean(content)
    score = SIA().polarity_scores(content)

    if score['neg'] > score['pos']: return -1
    elif score['pos'] > score['neg']: return 1
    else: return 0

    '''
        VADER:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
        print(sia().polarity_scores(samplestr))

        TEXTBLOB: 
        tb = TextBlob(content)
        tb.sentiment.polarity
    '''
import re
from nltk.tokenize import word_tokenize
from TurkishStemmer import TurkishStemmer
with open('turkish_stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = [line.strip() for line in file.readlines()]
stemmer = TurkishStemmer()
def temizle(yorum):
    # Küçük harfe çevir
    yorum = yorum.lower()
    # Özel karakterleri kaldır (<, >, #, ! vs.)
    yorum = re.sub(r'[^\w\s]', '', yorum)
    # Tokenize yap
    tokens = word_tokenize(yorum)
    # Stopwords ayıkla
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming uygula
    tokens = [stemmer.stem(word) for word in tokens]
    # Temizlenmiş veriyi döndür
    return tokens
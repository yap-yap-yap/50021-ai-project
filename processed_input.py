import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, regexp_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

WEBSITE_WORDS = ['http', 'twitter', 'com', 'pic', 'co']
MONTH_WORDS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'novemeber', 'december']


class ProcessedInput:
    def __init__(self, headline="Emtpy headline", article="Empty article"):
        self.article = article
        self.headline = headline
        self.banned_words = stopwords.words('english') + WEBSITE_WORDS + MONTH_WORDS
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

        vocab = self.read_vocab()
        vectorizer = CountVectorizer(preprocessor=self.dummy,
                                    tokenizer=self.dummy,
                                    vocabulary=vocab)
        tf_transformer = TfidfTransformer(use_idf=False)
        tfidf_transformer = TfidfTransformer(use_idf=True)

        self.article = self.tokenizer.tokenize(self.article.lower())
        self.article = self.remove_stopwords(self.article)
        self.article = self.lemmatize_tokens(self.article)
        self.article_count = vectorizer.fit_transform([self.article])
        self.article_tfidf = tfidf_transformer.fit_transform(self.article_count)
        self.article_tf = tf_transformer.fit_transform(self.article_count)

        self.headline = self.tokenizer.tokenize(self.headline.lower())
        self.headline = self.remove_stopwords(self.headline)
        self.headline = self.lemmatize_tokens(self.headline)
        self.headline_count = vectorizer.fit_transform([self.headline])
        self.headline_tfidf = tfidf_transformer.fit_transform(self.headline_count)
        self.headline_tf = tf_transformer.fit_transform(self.headline_count)

        self.cosine = cosine_similarity(self.headline_tfidf.toarray(), 
                                        self.article_tfidf.toarray())

        self.feature = np.concatenate([self.headline_tf.toarray(),
                                       self.cosine,
                                       self.article_tf.toarray()],
                        axis=1)
        print(f"shape: {self.feature.shape}")
    
    def dummy(self, doc):
        return doc

    def remove_stopwords(self, string):
        output = []
        for word in string:
            if word not in self.banned_words:
                output.append(word)
        return output
    
    def lemmatize_tokens(self, string):
        output = []
        for word in string:
            output.append(self.lemmatizer.lemmatize(word))
        return output

    def read_vocab(self):
        vocab_list = []
        with open("./vocab.txt", "r", encoding="ISO-8859-1") as f:
            for x in f:
                if x != "":
                    vocab_list.append(x.strip())
            return vocab_list
    
    def get_feature(self):
        return self.feature
        

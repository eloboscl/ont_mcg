import multiprocessing

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


def perform_topic_modeling(texts, num_topics=10, workers=None):
    if workers is None:
        workers = multiprocessing.cpu_count() - 1

    # Create Dictionary
    dictionary = Dictionary(texts)
    
    # Create Corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             per_word_topics=True,
                             workers=workers)
    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    return lda_model, coherence_lda
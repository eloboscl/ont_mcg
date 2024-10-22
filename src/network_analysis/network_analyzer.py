import multiprocessing
from collections import Counter
from functools import partial

import networkx as nx


def process_document(doc):
    authors = doc['authors']
    return list(itertools.combinations(authors, 2))

def create_author_collaboration_network(documents):
    # Parallel processing of documents
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        all_collaborations = pool.map(process_document, documents)
    
    # Flatten the list of collaborations
    all_collaborations = [collab for doc_collabs in all_collaborations for collab in doc_collabs]
    
    # Count collaborations
    collaboration_counts = Counter(all_collaborations)
    
    # Create graph
    G = nx.Graph()
    for (author1, author2), weight in collaboration_counts.items():
        G.add_edge(author1, author2, weight=weight)
    
    return G
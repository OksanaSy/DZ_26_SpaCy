import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

corpus = [
    "Physics is the natural science that studies matter, its motion and behavior through space and time.",
    "The political system in many countries is based on the principles of democracy.",
    "Economic growth is an increase in the production of economic goods and services, compared from one period of time to another.",
    "Social media platforms like Twitter and Facebook have a significant impact on public opinion.",
    "Literature offers insight into the human condition and allows us to explore different worlds and experiences.",
    "Artificial intelligence and machine learning are rapidly evolving fields with a wide range of applications.",
    "Climate change is a critical issue that affects global weather patterns and sea levels.",
    "Healthcare systems around the world are working to improve patient outcomes and reduce costs.",
    "The rise of electric vehicles represents a shift towards more sustainable transportation options.",
    "Space exploration continues to push the boundaries of human knowledge and technology.",
    "Modern art encompasses a wide range of styles and mediums, reflecting diverse cultural perspectives.",
    "Quantum computing has the potential to revolutionize various industries by solving complex problems more efficiently.",
    "The internet has transformed the way people communicate, access information, and conduct business.",
    "Cultural heritage sites are important for preserving history and fostering a sense of identity.",
    "Renewable energy sources, such as solar and wind power, are essential for reducing carbon emissions."
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

processed_corpus = [preprocess_text(doc) for doc in corpus]

tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(processed_corpus)]


model = Doc2Vec(vector_size=20, window=2, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

vectors = np.array([model.infer_vector(doc) for doc in processed_corpus])

def optimal_number_of_clusters(vectors, max_clusters=10):
    best_num_clusters = 2
    best_silhouette = -1
    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(vectors)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(vectors, labels)
        print(f"Кількість кластерів: {num_clusters}, Оцінка силуету: {silhouette_avg}")
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_num_clusters = num_clusters
    return best_num_clusters

num_clusters = optimal_number_of_clusters(vectors)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vectors)

labels = kmeans.labels_

for i, label in enumerate(labels):
    print(f"Документ {i+1}: '{corpus[i]}' належить до кластеру {label}")

cluster_range = range(2, 11)
scores = [silhouette_score(vectors, KMeans(n_clusters=k, random_state=42).fit(vectors).labels_) for k in cluster_range]

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, scores, marker='o')
plt.title('Оцінка силуету для різної кількості кластерів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Оцінка силуету')
plt.grid(True)
plt.show()


"""
Кількість кластерів: 2, Оцінка силуету: 0.1991037279367447
Кількість кластерів: 3, Оцінка силуету: 0.11848612874746323
Кількість кластерів: 4, Оцінка силуету: 0.10751821100711823
Кількість кластерів: 5, Оцінка силуету: 0.10239874571561813
Кількість кластерів: 6, Оцінка силуету: 0.07958648353815079
Кількість кластерів: 7, Оцінка силуету: 0.09801311790943146
Кількість кластерів: 8, Оцінка силуету: 0.06989095360040665
Кількість кластерів: 9, Оцінка силуету: 0.07515149563550949
Кількість кластерів: 10, Оцінка силуету: 0.06610651314258575
Документ 1: 'Physics is the natural science that studies matter, its motion and behavior through space and time.' належить до кластеру 0
Документ 2: 'The political system in many countries is based on the principles of democracy.' належить до кластеру 0
Документ 3: 'Economic growth is an increase in the production of economic goods and services, compared from one period of time to another.' належить до кластеру 1
Документ 4: 'Social media platforms like Twitter and Facebook have a significant impact on public opinion.' належить до кластеру 1
Документ 5: 'Literature offers insight into the human condition and allows us to explore different worlds and experiences.' належить до кластеру 1
Документ 6: 'Artificial intelligence and machine learning are rapidly evolving fields with a wide range of applications.' належить до кластеру 0
Документ 7: 'Climate change is a critical issue that affects global weather patterns and sea levels.' належить до кластеру 1
Документ 8: 'Healthcare systems around the world are working to improve patient outcomes and reduce costs.' належить до кластеру 0
Документ 9: 'The rise of electric vehicles represents a shift towards more sustainable transportation options.' належить до кластеру 1
Документ 10: 'Space exploration continues to push the boundaries of human knowledge and technology.' належить до кластеру 0
Документ 11: 'Modern art encompasses a wide range of styles and mediums, reflecting diverse cultural perspectives.' належить до кластеру 0
Документ 12: 'Quantum computing has the potential to revolutionize various industries by solving complex problems more efficiently.' належить до кластеру 1
Документ 13: 'The internet has transformed the way people communicate, access information, and conduct business.' належить до кластеру 1
Документ 14: 'Cultural heritage sites are important for preserving history and fostering a sense of identity.' належить до кластеру 0
Документ 15: 'Renewable energy sources, such as solar and wind power, are essential for reducing carbon emissions.' належить до кластеру 1

Найкращий силуетний коефіцієнт досягається при 2 кластерах.
"""

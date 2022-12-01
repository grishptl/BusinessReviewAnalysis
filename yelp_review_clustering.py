import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import textblob as tb
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Import WordCloud and STOPWORDS
from sklearn import metrics
from wordcloud import WordCloud
from wordcloud import STOPWORDS


print("=========Starting Data cleaning=========")

print("=========Read Business Reviews merged Dataset========")
# Read Business Reviews merged Dataset
review_df = pd.read_csv('Dataset/business_reviews.csv')

# Word Stemming
"""
i.e. Form a tree of words with the same meaning, where root would be the main keyword. 
look (root)
/          \
looking    looked   
"""

# Tokeninze
"""
also tokenize the tokens so we're only getting words, including those with apostrophes. Below is a function to pass through as an argument in the TfidfVectorizer to override the tokenizing and to add the stemming.
"""
print("=========Tokenizing and Word Stemming using NLTK========")

snowball = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [snowball.stem(word) for word in tokenizer.tokenize(text.lower())]

# vectorize the words of the corpus 
""" 
Putting the same meaning words in one vector
(look, looked, looking)
"""
#this function returns tfidf matrix
print("=========tfidf matrix generation========")
def vectorize_reviews(reviews):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize, max_features = 1000)
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    return X, words

"""
Each cluster that kMeans finds is a general topic of the reviews as a whole 
and is represented by words or groups of words. 

Each dimension in the cluster center coordinates is the relative frequency for a word in that cluster. 
We can find the indices of the words with highest frequency in each cluster 
and these indices correspond to their respective word in the array of tokens.

That way we can take a look at the words that represent the clusters the most 
and get an idea of what the latent topics are.
"""

def print_clusters(business_id, K = 8, num_words = 10):
    business_df = review_df[review_df['business_id'] == business_id]
    business_name = business_df['name'].unique()[0]
    reviews = business_df['text'].values
    X, words = vectorize_reviews(reviews)
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' + business_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

print("=========Total number of unique businesses========")
print(review_df['business_id'].unique)

max_review_business = review_df['business_id'].mode()
print(max_review_business)

business_df = review_df[review_df['business_id'] == "4JNXUYY8wbaaDmk3BPzlWw"]
business_name = business_df['name'].unique()[0]

print("Shape of new dataframe",business_df.shape)
print(business_name)

# Let's work on "Mon Ami Gabi" business
bus_id = '4JNXUYY8wbaaDmk3BPzlWw'
business_df = review_df[review_df['business_id'] == bus_id]
sns.countplot(x = business_df['stars'])

# Randomly choosing k = 5 and let's take top 12 words
print_clusters(bus_id, K = 5, num_words = 12)


# Let's also add ngrams to understand meaning better and TFIDF Vectorizing and saving file to csv
def save_tfidf_file(x_data, y_columns):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize, min_df = 0.0025, max_df = 0.05, max_features = 1000, ngram_range = (1, 3))
    count_array = x_data.toarray()
    df = pd.DataFrame(data=count_array,columns = y_columns)
    df.to_csv("Dataset/vectorizer_output.csv.gz", sep='\t', compression='gzip')
    print("vectorizer output file generated")

#--------NLP---------------
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

def build_word_corpus(category_name):
    print ('Building word corpus of %s category'%category_name)
    root=os.path.join(dir_path+'/Data/Categorized','dir_category_'+category_name.replace(' ','_').replace('/','_'))
    plain_text_reader=PlaintextCorpusReader(root,'.*')
    return plain_text_reader

def get_preprocessed_corpus_text(text_reader):
    print ('Removing the common stopwords')
    print ('Removing non alphanumeric words')
    print ('Converting all words into lowercase')
    s=stopwords.words()
    all_words=text_reader.words()
    t=nltk.Text(w.lower() for w in all_words if w.lower() not in s and w.isalpha())
    return t     

def plot_word_frequency_distribution(text,top=50):
    print ('Computing and plotting the frequency distribution for top %d words in the text'%top)
    fd=FreqDist(text)
    fd.plot(top)

def get_word_corpus_statistics(word_corpus):
    words=len(word_corpus.words())
    sentences=len(word_corpus.sents())
    characters=len(word_corpus.raw())
    unique_words=len(set(w.lower() for w  in word_corpus.words() if w.isalnum()))
    number_of_files=len(word_corpus.fileids())
    print ('Number of documents: ',number_of_files)
    print ('Number of characters: ',characters)
    print ('Number of sentences: ',sentences)
    print ('Number of words: ',words)
    print ('Number of unique words: ',unique_words)
  
def print_frequency_distribution():
    word_corpus = build_word_corpus('Chinese')
    print ('Chinese Restaturant Word Corpus Statistics')
    get_word_corpus_statistics(word_corpus)
    chinese_text=get_preprocessed_corpus_text(word_corpus)
    plot_word_frequency_distribution(chinese_text)  
#-----------------------  

#Unigram = 1, Bigram= 2, ...ngram to understand meaning in token of reviews
def vectorize_reviews2(reviews):
    vectorizer = TfidfVectorizer(tokenizer = tokenize, min_df = 5, max_df = 0.95, max_features = 8000, stop_words = 'english', ngram_range = (1, 3))
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    save_tfidf_file(x_data=X, y_columns = words)
    return X, words

def print_clusters2(business_id, K = 8, num_words = 10):
    business_df = review_df[review_df['business_id'] == business_id]
    business_name = business_df['name'].unique()[0]
    reviews = business_df['text'].values
    X, words = vectorize_reviews2(reviews)
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' +           business_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# Now check random K = 3 and num_words with ngram range from 1to 3
print_clusters2(bus_id, K = 5, num_words = 12)


# ### K- Means Model Evaluation Quality of Clusters - K-means
"""
The K-Means algorithm is one of the fastest clustering algorithms, but also one of the simplest:
First initialize  centroids randomly:  distinct instances are chosen randomly from the dataset 
and the centroids are placed at their locations.
Repeat until convergence (i.e., until the centroids stop moving):
Assign each instance to the closest centroid.
Update the centroids to be the mean of the instances that are assigned to them.
The KMeans class applies an optimized algorithm by default. 
To get the original K-Means algorithm (for educational purposes only), you must set init="random", n_init=1and algorithm="full". These hyperparameters will be explained below.
"""
# We will apply two evaluation methods, Inertia and Silhouette score.
# Find Optimal k values based on the evaluation.

# ### silhouette_score : to evaluate k-cluster
print("=========K- Means Model Evaluation Quality of Clusters - K-means========")
print("=========Silhouette_score : to evaluate k-cluster========")

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1, figsize=(16,9))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    f.savefig('SSEbyClusterCenterPlot.png')

reviews = business_df['text'].values
data, words = vectorize_reviews2(reviews)    
find_optimal_clusters(data, 20)

print("=========Elbow plot : to evaluate k-cluster========")

# Elbow plot to check the optimal K value
def elbow_plot(X, k_start, k_end):
    
    distortions = []
    K = range(k_start, k_end + 1)
    for k in K:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(K, distortions)
    plt.xticks(K)
    plt.title('Elbow curve')
    plt.savefig('ElbowCurve.png')

reviews = business_df['text'].values
X, words = vectorize_reviews2(reviews)
elbow_plot(X, 1, 20)


# We found that for silhouette_score plot K = 6 is the highest peek value and we will consider it as optimal k value
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7)
label = kmeans.fit(X)

print("=========K-means clustering with K = 7========")

#Simple clustering
clusters = MiniBatchKMeans(n_clusters=7, init_size=1024, batch_size=2048, random_state=20).fit_predict(X)

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    f.savefig('TSNE_Cluster_Plot.png')

plot_tsne_pca(X, clusters)

print("=========Top keywords per cluster========")
def get_top_keywords_per_clusters(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_top_keywords_per_clusters(X, clusters, words, 50)

# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# # Vectorize document using TF-IDF
tf_idf_vect = TfidfVectorizer(lowercase=True,  min_df = 5, max_df = 0.95, max_features = 8000, stop_words = 'english', ngram_range = (1, 3),                        tokenizer = tokenizer.tokenize)


reviews = company_df['text'].values
X, words = vectorize_reviews2(reviews)
# Fit and Transfrom Text Data
X_train_counts = tf_idf_vect.fit_transform(reviews)

# Check Shape of Count Vector
X_train_counts.shape

# Import KMeans Model
from sklearn.cluster import KMeans

# Create Kmeans object and fit it to the training data 
kmeans = KMeans(n_clusters=7).fit(X_train_counts)

# Get the labels using KMeans
pred_labels = kmeans.labels_

# Compute DBI score
dbi = metrics.davies_bouldin_score(X_train_counts.toarray(), pred_labels)

# Compute Silhoutte Score
ss = metrics.silhouette_score(X_train_counts.toarray(), pred_labels , metric='euclidean')

# Print the DBI and Silhoutte Scores
print("DBI Score: ", dbi, "\nSilhoutte Score: ", ss)


print(np.unique(pred_labels, return_counts=True))
#np.unique(pred_labels)

print("=========Saving word cloud per cluster========")

def word_cloud(text,wc_title,wc_file_name='wordcloud.jpeg'):
    # Create stopword list
    stopword_list = set(STOPWORDS) 

    # Create WordCloud 
    word_cloud = WordCloud(width = 800, height = 500, 
                           background_color ='white', 
                           stopwords = stopword_list, 
                           min_font_size = 14).generate(text) 

    # Set wordcloud figure size
    plt.figure(figsize = (8, 6)) 
    
    # Set title for word cloud
    plt.title(wc_title)
    
    # Show image
    plt.imshow(word_cloud) 

    # Remove Axis
    plt.axis("off")  

    # save word cloud
    plt.savefig(wc_file_name,bbox_inches='tight')

    # show plot
    plt.show()


df=pd.DataFrame({"text":reviews,"labels":pred_labels})
for i in df.labels.unique():
    new_df=df[df.labels==i]
    text="".join(new_df.text.tolist())
    word_cloud(text,"Cluster "+str(i), "cluster"+str(i)+'.jpeg')

company_df['cluster_labels'] = pred_labels
cluster0_data = company_df[company_df["cluster_labels"] == 0]

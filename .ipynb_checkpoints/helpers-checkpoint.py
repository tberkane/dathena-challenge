import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

translator = Translator()


def translate_chunk(to_translate, translated_list, src_language):
    """
    Translates to_translate using googletrans and then appends the result to translated_list.
    """
    translation = translator.translate(to_translate, src=src_language)
    translated_list.append(translation.text)


PUNCUATION_LIST = list(string.punctuation)

lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    """
    Returns the lemmatized version of text
    """
    return [lemmatizer.lemmatize(w) for w in text]


def get_doc_frequency(data):
    """
    Returns the frequency with which each word in the vocabulary occurs in the data
    """
    joined_data = " ".join(data)
    words = {w: set() for w in set(joined_data.split())}
    for i, recipe in enumerate(data):
        for w in recipe.split():
            if w in words:
                words[w].add(i)

    data_len = len([d for d in data if d != ''])
    words = sorted({w: len(words[w]) / data_len for w in words}.items(), key=lambda x: -x[1])
    return words


# function to replace words with n-grams in the text
def replace_ngram(text, bigrams):
    """
    Replaces words with n-grams in text
    """
    for gram in bigrams:
        text = text.replace(gram, '_'.join(gram.split()))
    return text


def preprocess_section(section, df_recipes, freq_thresh_upper=0.44, freq_thresh_lower=0.04, lemmatize=True,
                       bigrams=False):
    """
    Creates a new column in the dataframe containing the preprocessed version of a section
    :param section: section to preprocess, can be either "title", "ingredients", "direction" or "text"
    :param df_recipes: dataframe containing the columns to preprocess
    :param freq_thresh_upper: Word frequency threshold over which words will be removed (-> frequent words are removed)
    :param freq_thresh_lower: Word frequency threshold under which words will be removed (-> unfrequent words are removed)
    :param lemmatize: Whether to lemmatize the text
    :param bigrams: Whether to include bigrams
    """
    df_recipes[f"preprocessed_{section}"] = df_recipes[f"translated_{section}"].str.lower()
    for char in PUNCUATION_LIST:
        df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(
            lambda r: r.replace(char, " "))
    df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(word_tokenize)
    if lemmatize:
        df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(lemmatize_text)
    df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(
        # remove single characters and numbers
        lambda ts: [t for t in ts if len(t) > 1 and not t[0].isnumeric()])

    words = get_doc_frequency(
        list(df_recipes[f'preprocessed_{section}'].apply(lambda t: " ".join(t))))  # word frequencies
    stop_words = list(stopwords.words('english'))  # common english stopwords
    stop_words.extend([w[0] for w in words if w[1] >= freq_thresh_upper or w[
        1] <= freq_thresh_lower])  # add words occurring too frequently or too infrequently to stopwords
    df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(
        lambda ts: [t for t in ts if t not in stop_words])  # remove all stopwords

    df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(lambda t: " ".join(t))

    if bigrams:
        data_words = [[w for w in l.split()] for l in df_recipes[f'preprocessed_{section}'].to_list()]
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_documents(data_words)
        finder.apply_freq_filter(10)  # only keep bigrams which occur at least 10 times
        bigram_scores = finder.score_ngrams(bigram_measures.pmi)  # calculate Pointwise Mutual Information
        if not bigram_scores:
            print(f"{section}: No bigrams found!")
        else:
            bigram_pmi = pd.DataFrame(bigram_scores)
            bigram_pmi.columns = ['bigram', 'pmi']
            bigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)  # rank bigrams by PMI
            bigram_pmi = bigram_pmi[bigram_pmi.pmi > 6]  # only keep bigrams which make sense
            bigrams = [' '.join(x) for x in bigram_pmi.bigram.values]
            df_recipes[f'preprocessed_{section}'] = df_recipes[f'preprocessed_{section}'].apply(replace_ngram,
                                                                                                args=(bigrams,))


def generate_wordcloud(data, title, bigrams=True):
    """
    Generates a wordcloud showing the most frequent words in the data.
    """
    wc = WordCloud(width=400, height=330, max_words=50, background_color="white", collocations=bigrams).generate(data)
    plt.figure(figsize=(12, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=13)
    plt.show()


SECTIONS = ["title", "ingredients", "directions", "text"]


def visualize_projection(df_recipes, projection_method="pca"):
    """
    Projects the TF-IDF of each section to 2 dimensions and then plots the result
    :param df_recipes: dataframe containing the sections to project
    :param projection_method: algorithm to use for projection, can be either "pca" or "tsne"
    """
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    for i, section in enumerate(SECTIONS):
        # tfidf
        vectorizer = TfidfVectorizer()
        response = vectorizer.fit_transform(df_recipes[f'preprocessed_{section}'])
        tfidf = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names_out())
        if projection_method == "pca":
            pca = PCA(n_components=2).fit(tfidf)
            data2D = pca.transform(tfidf)
        elif projection_method == "tsne":
            data2D = manifold.TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(tfidf)
        else:
            print("Unknown method")
            return

        # plot
        sbplt = ax[i // 2, i % 2]
        sbplt.set_title(section.capitalize(), fontsize=20)
        sbplt.scatter(data2D[:, 0], data2D[:, 1])
    if projection_method == "pca":
        fig.text(0.5, 0, 'Principal component 1', ha='center', fontsize=20)
        fig.text(0, 0.5, 'Principal component 2', va='center', rotation='vertical', fontsize=20)


def visualize_projection_interactive(section, df_recipes, projection_method="pca", clusters=[0]*150):
    """
    Projects the TF-IDF of a section to 2 dimensions and then displays an interactive plot of the result
    :param section: name of the section to visualize
    :param df_recipes: dataframe containing the sections to project
    :param projection_method: algorithm to use for projection, can be either "pca" or "tsne"
    """
    # tfidf
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(df_recipes[f'preprocessed_{section}'])
    tfidf = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names_out())
    if projection_method == "pca":
        pca = PCA(n_components=2).fit(tfidf)
        data2D = pca.transform(tfidf)
    elif projection_method == "tsne":
        data2D = manifold.TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(tfidf)
    else:
        print("Unknown method")
        return
    
    palette = (
    "#000000", "#009292", "#ff6db6", "#490092", "#006ddb", "#b66dff", "#6db6ff", "#b6dbff", "#920000", "#24ff24","#db6d00","#ffff6d")

    p = figure(title=section.capitalize(), height=700, width=1100, tooltips=[("Title", "@title")])
    colors = [palette[p] for p in clusters]
    titles = list(df_recipes.translated_title)
    source = ColumnDataSource(data=dict(x=data2D[:, 0], y=data2D[:, 1], color=colors, title=titles))
    p.circle(x="x", y="y", source=source, size=12, color="color", line_color=None, fill_alpha=1)

    show(p)  # show the results

    
def get_clusters(section, df_recipes, num_clusters=2, clustering_method="kmeans", eps=10, min_samples=20):
    # tf-idf
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(df_recipes[f"preprocessed_{section}"])
    X = response.toarray()
    # standardize data before k-means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if clustering_method=="kmeans":
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=5000).fit(X_scaled)
        clusters = kmeans.predict(X_scaled)
    elif clustering_method=="dbscan":
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    else:
        print("Unknown method")
        return []
    return clusters
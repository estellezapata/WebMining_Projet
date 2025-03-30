
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import re
import string
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


nltk.download('stopwords')
nltk.download('wordnet')

bd_clean = pd.read_csv('bd_clean.csv')

# Import data
# Lire les trois CSV
# df1 = pd.read_csv('bd_clean_part1.csv')
# df2 = pd.read_csv('bd_clean_part2.csv')
# df3 = pd.read_csv('bd_clean_part3.csv')

# # Concatenation des DataFrames
# bd_clean = pd.concat([df1, df2, df3], ignore_index=True)

# Import data for tweet embedding
bd_tweet_embedding = pd.read_csv('df_tweet_embedding.csv')
bd_tweet_embedding = bd_tweet_embedding.dropna(subset=['dbscan_cluster', 'cluster_name'])
bd_tweet_embedding['dbscan_cluster'] = bd_tweet_embedding['dbscan_cluster'].astype(int)  

# --- TF-IDF --------------------------
# Preparation of tweets
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Minuscule
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Supprimer URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Supprimer mentions et hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer ponctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Stopwords + lemmatisation
    return " ".join(words)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Web Mining Project - Guillemard - Pommier - Liong - Zapata",
    page_icon="📊",
    layout="wide"
)

# --- En-tête ---
st.markdown(
    """
    <style>
        .header-container {
            width: 90vw;
            background-color: #2C3E50;
            padding: 20px 0;
            text-align: center;
        }
        .header-title {
            font-size: 32px;
            color: white;
            font-weight: bold;
        }
        .content-box {
            background-color: #2C3E50;
            padding: 6px;
            border-radius: 10px;
            margin-top: 20px;
            margin-down: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding-left: 30px;
        }
        
        
    </style>
    <div class="header-container">
        <span class="header-title">MINING KNOWLEDGE FROM SOCIAL MEDIA DATA DURING CRISIS EVENTS</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Barre latérale (menu déroulant) ---
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select an Event:", 
                            ["General Analysis", 'wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing'])
st.sidebar.markdown("""<div style="margin-bottom: 50px;"></div>""", unsafe_allow_html=True)


# # --- Barre latérale : Sélection du cluster ---
# st.sidebar.title("Tweet Embeddings : Select a Cluster")
# cluster_list = bd_tweet_embedding["cluster_name"].unique().tolist()
# selected_cluster = st.sidebar.selectbox("Select a Cluster:", cluster_list)

# --- General Analysis ---
if menu == "General Analysis":
    # st.subheader("📊 Statistiques Générales")
    # st.markdown(
    # """
    # <style>
    #     .header {
    #         background-color: #2C3E50;
    #         padding: 15px;
    #         text-align: center;
    #         font-size: 24px;
    #         color: white;
    #         font-weight: bold;
    #         border-radius: 10px;
    #     }
    # </style>
    # <div class="header">📊 Statistiques Générales</div>
    # """,
    # unsafe_allow_html=True
    # )
    st.markdown(
        """
        <div class="content-box">
            <h3>📊 General Analysis</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Calcul des statistiques générales
    nb_users_uniques = bd_clean['node_id_user'].nunique()
    nb_tweets_uniques = bd_clean['node_id_tweet'].nunique()
    nb_event_types = bd_clean['eventType_event'].nunique()
    types_event = bd_clean['eventType_event'].unique()
    nb_events = bd_clean['id_event_y'].nunique()
    events_uniques = bd_clean['id_event_y'].unique()
    avg_followers = bd_clean.groupby('node_id_user')['followers_count_user'].mean().mean()
    avg_followers = round(avg_followers)

    # Affichage des statistiques générales
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Number of Users", value=f"{nb_users_uniques:,}")
    col2.metric(label="Total Number of Tweets", value=f"{nb_tweets_uniques:,}")
    col3.metric(label="Average Number of Followers per User", value=f"{avg_followers:,}")

    col1.metric(label="Number of Event Categories", value=f"{nb_event_types:,}")
    col2.metric(label="Number of Events", value=f"{nb_events:,}")

    # st.write(f"Types d'événements uniques : {types_event}")
    # st.write(", ".join(types_event))
    st.write(f"List of Event Categories:")
    # Création d'un DataFrame pour afficher les types d'événements dans un tableau
    df_types_event = pd.DataFrame(types_event, columns=["Event Categories"])
    st.table(df_types_event)  # Affichage du tableau

    # st.write(f"ID d'événements uniques : {events_uniques}")
    # st.write(", ".join(events_uniques.astype(str)))
    st.write(f"List of Events:")
    # Création d'un DataFrame pour afficher les types d'événements dans un tableau
    df_id_event = pd.DataFrame(events_uniques, columns=["Events"])
    st.table(df_id_event)  # Affichage du tableau

    # st.write(f"Nombre moyen de followers par utilisateur unique : {avg_followers:.2f}")

# --- Statistiques pour un type d'événement spécifique ---
elif menu in ['wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing']:
    # st.subheader(f"{menu.capitalize()} - Statistiques")
    st.markdown(
        f"""
        <div class="content-box">
            <h3>Analysis of {menu.capitalize()}-Related Events</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Filtrage des tweets par type d'événement
    event_tweets = bd_clean[bd_clean['eventType_event'] == menu]

    # Calcul des statistiques pour chaque événement
    nb_users_event = event_tweets['node_id_user'].nunique()
    nb_tweets_event = event_tweets['node_id_tweet'].nunique()
    tweets_retweeted = event_tweets[event_tweets['retweet_count_tweet'] > 0]['node_id_tweet'].nunique()
    pourcentage_retweet = (tweets_retweeted / nb_tweets_event) * 100
    total_retweets = event_tweets['retweet_count_tweet'].sum()
    tweet_max_retweet = event_tweets.loc[event_tweets['retweet_count_tweet'].idxmax()]
    texte_tweet_max_retweet = tweet_max_retweet['text_tweet']
    tweets_liked = event_tweets[event_tweets['favorite_count_tweet'] > 0]['node_id_tweet'].nunique()
    pourcentage_likes = (tweets_liked / nb_tweets_event) * 100
    total_likes = event_tweets['favorite_count_tweet'].sum()
    avg_followers_event = event_tweets.groupby('node_id_user')['followers_count_user'].mean().mean()
    avg_followers_event = round(avg_followers_event)
    top_3_users = event_tweets.groupby('screen_name_user')['node_id_tweet'].nunique().nlargest(3)
    priority_percentages = event_tweets.groupby('annotation_postPriority_tweet')['node_id_tweet'].nunique() / nb_tweets_event * 100
    # top_5_hashtags = event_tweets.groupby('id_hashtag_y')['occurences_hashtag'].sum().nlargest(5)
    top_5_hashtags = (
        event_tweets.groupby('id_hashtag_y')['occurences_hashtag']
        .sum()
        .nlargest(5)
        .reset_index()  # Convertir en DataFrame pour renommer les colonnes
    )
    # Renommer les colonnes
    top_5_hashtags = top_5_hashtags.rename(columns={
        'id_hashtag_y': 'Hashtag',
        'occurences_hashtag': 'Number of occurrences'
    })
    # Appliquer le formatage anglais (virgules pour les milliers)
    top_5_hashtags['Number of occurrences'] = top_5_hashtags['Number of occurrences'].apply(lambda x: f"{x:,}")


    ## The (temporal) distribution of tweet ------------------
    # Define date
    # date_range = pd.date_range(event_tweets['date'].min(), event_tweets['date'].max(), freq='D')
    # # Number of tweets per day
    # tweet_count_per_day = event_tweets.groupby('date').size()
    # # Réindexer pour inclure toutes les dates avec des 0 là où il n'y a pas de tweets
    # tweet_count_per_day = tweet_count_per_day.reindex(date_range, fill_value=0)
    # fig_distrib_tweet = px.line(tweet_count_per_day, x="Date", y="Number of tweets", title=f"Tweet distribution of {menu}")



    # Création du graphique à barres
    # fig, ax = plt.subplots(figsize=(8, 6))  # Taille du graphique
    # priority_percentages.plot(kind='bar', ax=ax, color='skyblue')  # Création du bar plot
    # ax.set_title("Pourcentage de chaque valeur de 'annotation_postPriority_tweet'", fontsize=14)  # Titre
    # ax.set_xlabel('Annotation Post Priority', fontsize=12)  # Label de l'axe X
    # ax.set_ylabel('Pourcentage (%)', fontsize=12)  # Label de l'axe Y
    # ax.set_xticklabels(priority_percentages.index, rotation=45, ha="right")  # Rotation des labels de l'axe X

    # Conversion en DataFrame pour Plotly
    df_priority = pd.DataFrame(priority_percentages).reset_index()
    df_priority.columns = ['Catégorie', 'Valeur']

    # Création du graphique à barres avec Plotly
    # fig = px.bar(df_priority, x="Catégorie", y="Valeur", title="Share of Published Priority Tweets",
    #             labels={'Catégorie': ' Annotation Post Priority ', 'Valeur': ' Pourcentage (%) '},
    #             color="Catégorie",  # Couleur différente pour chaque barre
    #             color_continuous_scale='Blues')  # Choix d'une palette de couleurs
    fig = px.bar(
        df_priority, 
        x="Catégorie", 
        y="Valeur", 
        title="Share of Published Priority Tweets",
        labels={'Catégorie': 'Annotation Post Priority', 'Valeur': 'Pourcentage (%)'},
        color="Catégorie",  # Couleur différente pour chaque barre
        color_continuous_scale='Blues'
    )
    fig.update_traces(hovertemplate='%{y:.2f}%')

    # Affichage des résultats pour chaque événement
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Number of Tweets", value=f"{nb_tweets_event:,}")
    col2.metric(label="Number of Retweeted Tweets", value=f"{tweets_retweeted:,} ({pourcentage_retweet:.2f}%)")
    col3.metric(label="Total Number of Retweets", value=f"{total_retweets:,}")

    # st.write(f"5. Le tweet le plus retweeté : {texte_tweet_max_retweet} avec {tweet_max_retweet['retweet_count_tweet']} retweets")
    # Affichage stylisé avec un encadré markdown
    # st.subheader("Most Retweeted Tweet")
    st.write(f"Most Retweeted Tweet:")
    st.markdown(
        f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            background-color: #e3e6eb;
            border-left: 6px solid #1DA1F2;
            font-size: 16px;
            color: #222;  /* Texte plus foncé */
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        ">
            <strong style="color: #1DA1F2;">{tweet_max_retweet['retweet_count_tweet']} retweets</strong><br>
            <em>{texte_tweet_max_retweet}</em>
        </div>
        """, unsafe_allow_html=True
    )

    # st.write(f"   Texte du tweet le plus retweeté : {texte_tweet_max_retweet}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Number of Users", value=f"{nb_users_event:,}")
    col2.metric(label="Average Number of Followers per User", value=f"{avg_followers_event:,}")
    col1.metric(label="Number of Liked Tweets", value=f"{tweets_liked:,} ({pourcentage_likes:.2f}%)")
    col4.metric(label="Total Number of Likes", value=f"{total_likes:,}")

    # st.write(f"6. Nombre de tweets avec au moins un like : {tweets_liked} ({pourcentage_likes:.2f}%)")
    st.write(f"Top 3 Users with the Most Tweets:")
    st.write(top_3_users)
    # st.write(f"10. Pourcentage de chaque valeur de 'annotation_postPriority_tweet' :")
    # st.write(priority_percentages)

    # Affichage du graphique dans Streamlit
    # st.pyplot(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Top 5 Most Frequent Hashtags and Their Occurrences:")
    st.write(top_5_hashtags)

    # st.plotly_chart(fig_distrib_tweet)
    # st.line_chart(tweet_count_per_day.set_index("Date"))
    # st.area_chart(tweet_count_per_day.set_index("Date"))
    
    # --- 📊 Graphique de distribution des tweets dans le temps ---
    # st.subheader(f"📈 Distribution des tweets pour {menu.capitalize()}")
    st.markdown(
            f"""
            <div class="content-box">
                <h3>📈 Time Distribution of Published Tweets for the {menu.capitalize()} event</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Définir la plage de dates complète
    event_tweets['date'] = pd.to_datetime(event_tweets['date'])  # S'assurer que la colonne date est bien en datetime
    event_tweets['word_counts'] = event_tweets['text_tweet'].apply(lambda x: len(x.split()))

    date_range = pd.date_range(event_tweets['date'].min(), event_tweets['date'].max(), freq='D')

    # Compter les tweets par jour pour cet événement
    # tweet_count_per_day = event_tweets.groupby('date').size()
    # Réindexer pour inclure toutes les dates avec des 0 là où il n'y a pas de tweets
    # tweet_count_per_day = tweet_count_per_day.reindex(date_range, fill_value=0)
    
    # Compter les tweets par jour pour ce type d'événement
    # tweet_count_per_day = df_flood.groupby('date').size()
    tweet_count_per_day = event_tweets.groupby(['date', 'id_event_y']).size().reset_index(name='count')

    # Réindexer pour inclure toutes les dates avec des 0 là où il n'y a pas de tweets
    # tweet_count_per_day = tweet_count_per_day.reindex(date_range, fill_value=0)
    tweet_pivot = tweet_count_per_day.pivot(index='date', columns='id_event_y', values='count').fillna(0)
    tweet_pivot = tweet_pivot.reindex(date_range, fill_value=0)
    
    # Compter les mots des tweets par jour pour ce type d'événement
    word_count_per_day = event_tweets.groupby(['date', 'id_event_y'])['word_counts'].sum().reset_index(name='count')

    # # Réindexer pour inclure toutes les dates avec des 0 là où il n'y a pas de tweets
    word_pivot = word_count_per_day.pivot(index='date', columns='id_event_y', values='count').fillna(0)
    word_pivot = word_pivot.reindex(date_range, fill_value=0)
    
    # Création du graphique interactif
    fig = go.Figure()
    # graph simple
    # fig.add_trace(go.Scatter(
    #     x=tweet_count_per_day.index,
    #     y=tweet_count_per_day.values,
    #     mode='lines',
    #     line=dict(color='blue'),
    #     marker=dict(size=6),
    #     hoverinfo='x+y',
    #     name=menu.capitalize()
    # ))

    # fig.update_layout(
    #     title=f"Évolution des Tweets pour {menu.capitalize()}",
    #     xaxis_title="Date",
    #     yaxis_title="Nombre de Tweets",
    #     template="plotly_white"
    # )
    for event_id in tweet_pivot.columns:
        fig.add_trace(go.Scatter(
            x=tweet_pivot.index,
            y=tweet_pivot[event_id],
            mode='lines',  
            line=dict(width=2),
            marker=dict(size=6),
            hoverinfo='x+y',
            name=f'Event {event_id}',
            hovertemplate='%{x}<br>%{y} Tweets' if any(tweet_pivot[event_id] != 0) else None

        ))

    # Mise en forme du graphique
    fig.update_layout(
        title=f"Time Distribution of Published Tweets - {menu}",
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )
    # Création du graphique interactif
    fig = go.Figure() 
    for event_id in tweet_pivot.columns:
        fig.add_trace(go.Scatter(
            x=tweet_pivot.index,
            y=tweet_pivot[event_id],
            mode='lines',  
            line=dict(width=2),
            marker=dict(size=6),
            hoverinfo='x+y',
            name=f'Event {event_id}',
            hovertemplate='%{x}<br>%{y} Tweets' if any(tweet_pivot[event_id] != 0) else None

        ))

    # Mise en forme du graphique
    fig.update_layout(
        title=f"Time Distribution of Published Tweets - {menu}",
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Afficher le graphique
    st.plotly_chart(fig)

    # Distribution des mots -----------------
    fig_word = go.Figure()

    for event_id in word_pivot.columns:
        fig_word.add_trace(go.Scatter(
            x=word_pivot.index,
            y=word_pivot[event_id],
            mode='lines',  
            line=dict(width=2),
            marker=dict(size=6),
            hoverinfo='x+y',
            name=f'Event {event_id}',
            hovertemplate='%{x}<br>%{y} Tweets' if any(word_pivot[event_id] != 0) else None

        ))

    # Mise en forme du graphique
    fig_word.update_layout(
        title=f"Time Distribution of Published Word Count - {menu}",
        xaxis_title="Date",
        yaxis_title="Number of Words in Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )
    # Création du graphique interactif
    fig_word = go.Figure() 
    for event_id in word_pivot.columns:
        fig_word.add_trace(go.Scatter(
            x=word_pivot.index,
            y=word_pivot[event_id],
            mode='lines',  
            line=dict(width=2),
            marker=dict(size=6),
            hoverinfo='x+y',
            name=f'Event {event_id}',
            hovertemplate='%{x}<br>%{y} Words' if any(word_pivot[event_id] != 0) else None

        ))

    # Mise en forme du graphique
    fig_word.update_layout(
        title=f"Time Distribution of Published Word Count - {menu}",
        xaxis_title="Date",
        yaxis_title="Number of Words in Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Afficher le graphique
    st.plotly_chart(fig_word)

    ## TF-IDF ---------------
    # --- Paramètres interactifs ---
    st.sidebar.subheader("TF-IDF Analysis Settings")
    st.sidebar.markdown("*You can adjust the TF-IDF analysis settings and see the changes in the chart.*")
    top_k = st.sidebar.slider("Number of Words to Display", min_value=5, max_value=5, value=50, step=10)
    num_clusters = st.sidebar.slider("Number of K-Means Clusters", min_value=2, max_value=5, value=5)
    st.sidebar.markdown("""<div style="margin-bottom: 10px;"></div>""", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="content-box">
            <h3>TF-IDF for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    # Prétraitement des tweets pour l'événement sélectionné
    tweets_cleaned = [preprocess_text(tweet) for tweet in event_tweets['text_tweet']]

    # 3️⃣ Calculer le TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(tweets_cleaned)
    vocab = vectorizer.get_feature_names_out()

    # 4️⃣ Trouver les k mots avec le plus haut IDF
    idf_scores = vectorizer.idf_
    # top_k = 50  # Nombre de mots importants à extraire
    top_k_indices = idf_scores.argsort()[-top_k:]  # Indices des k plus grands IDF
    top_k_words = [vocab[i] for i in top_k_indices]

    # 5️⃣ Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, init="random", learning_rate=200)
    tfidf_2d = tsne.fit_transform(tfidf_matrix.T[top_k_indices].toarray())  # Transposer pour obtenir les vecteurs de mots

    # 6️⃣ Clustering avec K-Means
    # num_clusters = 5  # Ajuste selon tes besoins
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_2d)

    # 7️⃣ Création du DataFrame pour Plotly
    # df_tfidf_graph = pd.DataFrame({
    #     'word': top_k_words,
    #     'x': tfidf_2d[:, 0],
    #     'y': tfidf_2d[:, 1],
    #     'cluster': clusters
    # })

    # # 8️⃣ Définition des couleurs pour les clusters (en utilisant une palette discrète)
    # cluster_colors = px.colors.qualitative.Set1  # Palette de couleurs discrètes

    
    # 8️⃣ Visualisation interactive avec Plotly
    # fig_tfidf = px.scatter(df_tfidf_graph, x='x', y='y', text='word', color='cluster', title=f"Clustering des mots importants pour {menu.capitalize()}",
    #                  labels={'x': 't-SNE X', 'y': 't-SNE Y'}, hover_data={'word': True, 'cluster': True})
    
    # # Affichage dans Streamlit
    # st.plotly_chart(fig_tfidf)
    # 7️⃣ Visualisation des clusters
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(top_k_words):
        x, y = tfidf_2d[i, 0], tfidf_2d[i, 1]
        plt.scatter(x, y, c=f"C{clusters[i]}", s=100)  # Couleur selon le cluster
        plt.text(x + 0.1, y + 0.1, word, fontsize=12)

    plt.title(f"Word Clustering with the Highest TF-IDF Scores for the {menu.capitalize()} event")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.grid(True)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    ###  Word embedding -------------------------------------
    st.markdown(
        f"""
        <div class="content-box">
            <h3>Word embedding for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    # st.subheader(f"📈 Word embedding for {menu.capitalize()}")
    image_path = os.path.join("pictures", f"Word_Embeddings-{menu}.png")
    st.image(image_path, caption=f"Word Embeddings - {menu.capitalize()}", use_container_width=True)

    ###  Tweet embedding -------------------------------------
    # st.subheader(f"📈 Tweet embedding for {menu.capitalize()}")
    st.markdown(
        f"""
        <div class="content-box">
            <h3>Tweet embedding for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    # Filtrage des tweets par type d'événement
    tweet_embed_event = bd_tweet_embedding[bd_tweet_embedding['eventType'] == menu]
    # 🔹 **Liste des clusters disponibles pour cet événement**
    available_clusters = tweet_embed_event["cluster_name"].unique().tolist()

    # --- Barre latérale : Sélection du cluster (Mise à jour dynamique) ---
    if available_clusters:
        st.sidebar.subheader("Tweet Embeddings")
        selected_cluster = st.sidebar.selectbox("Select a cluster:", available_clusters)
        
        # --- Filtrage des tweets selon le cluster sélectionné ---
        filtered_tweets = tweet_embed_event[tweet_embed_event["cluster_name"] == selected_cluster][["text"]]

        # --- Affichage des tweets ---
        st.subheader(f"📌 Tweets of cluster : {selected_cluster}")

        if not filtered_tweets.empty:
            sampled_tweets = filtered_tweets.sample(n=min(10, len(filtered_tweets)), random_state=42)

            for i, tweet in enumerate(sampled_tweets["text"], 1):
                # st.write(f"**{i}.** {tweet}")
                st.markdown(
                    f"""
                    <div style="
                        padding: 15px;
                        border-radius: 10px;
                        background-color: #e3e6eb;
                        border-left: 6px solid #1DA1F2;
                        font-size: 16px;
                        color: #222;  /* Texte plus foncé */
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                    ">
                        <strong style="color: #1DA1F2;">{i}</strong><br>
                        <em>{tweet}</em>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown("""<div style="margin-bottom: 5px;"></div>""", unsafe_allow_html=True)
        else:
            st.warning("Aucun tweet disponible pour ce cluster.")
    else:
        st.warning(f"Aucun cluster disponible pour {menu.capitalize()}.")

    # # --- Filtrage des tweets selon le cluster sélectionné ---
    # filtered_tweets = event_tweets[event_tweets["cluster_name"] == selected_cluster][["text"]]
    # # --- Affichage des tweets ---
    # st.subheader(f"📌 Tweets du cluster : {selected_cluster}")
    # if not filtered_tweets.empty:
    #     sampled_tweets = filtered_tweets.sample(n=min(10, len(filtered_tweets)), random_state=42)
        
    #     for i, tweet in enumerate(sampled_tweets["text"], 1):
    #         st.write(f"**{i}.** {tweet}")
    # else:
    #     st.warning("Aucun tweet disponible pour ce cluster.")




# --- Pied de page de la barre latérale ---
# st.sidebar.markdown(
#     """
#     <div class="sidebar-footer">
#         <p><strong>Created by:</strong><br>
#         GUILLEMARD Mattéo<br>
#         LIONG-WEE-KWONG Jade<br>
#         POMMIER-MAURUSSANE Clothilde<br>
#         ZAPATA Estelle</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# --- Auteur en bas de la barre latérale ---
for _ in range(3):  
    st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; font-size: 12px; color: gray; line-height: 0.2;">
        <div style="text-align: left; margin-right: 10px;">
            <p><strong>Created by:</strong></p>
            <p>GUILLEMARD Mattéo</p>
            <p>LIONG-WEE-KWONG Jade</p>
            <p>POMMIER-MAURUSSANE Clothilde</p>
            <p>ZAPATA Estelle</p>
        </div>
    """,
    unsafe_allow_html=True
)
# Utiliser st.sidebar.image avec le chemin absolu
logo_path = os.path.join(os.getcwd(), 'pictures', 'tse_logo.png')
st.sidebar.image(logo_path, width=120)


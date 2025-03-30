## This code create the Dashboard with streamlit

# 0 - Import libraries ------------------------------------------
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

# 1 - import data -----------------------------------------------
# Import data for VSCode
# bd_clean = pd.read_csv('bd_clean.csv')

# Import data for Git Hub
# Lire les trois CSV
df1 = pd.read_csv('bd_clean_part1.csv')
df2 = pd.read_csv('bd_clean_part2.csv')
df3 = pd.read_csv('bd_clean_part3.csv')

# Concatenation des DataFrames
bd_clean = pd.concat([df1, df2, df3], ignore_index=True)

# Import data for tweet embedding
bd_tweet_embedding = pd.read_csv('df_tweet_embedding.csv')
bd_tweet_embedding = bd_tweet_embedding.dropna(subset=['dbscan_cluster', 'cluster_name'])
bd_tweet_embedding['dbscan_cluster'] = bd_tweet_embedding['dbscan_cluster'].astype(int)  

# Preparation for TF-IDF --------------------------
# Preparation of tweets
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  
    text = re.sub(r'\@\w+|\#', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  
    return " ".join(words)

# --- Configuration de la page ---------------------
st.set_page_config(
    page_title="Web Mining Project - Guillemard - Pommier - Liong - Zapata",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .header-container {
            width: 90vw;
            background-color: #061987;
            padding: 20px 0;
            text-align: center;
        }
        .header-title {
            font-size: 32px;
            color: white;
            font-weight: bold;
        }
        .content-box {
            background-color: #A8B9E7;
            padding: 6px;
            border-radius: 10px;
            margin-top: 20px;
            margin-down: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding-left: 30px;
        }
        .subtitle-box {
            background-color: #8A9FE0;
            padding: 6px;
            border-radius: 10px;
            margin-top: 20px;
            margin-down: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding-left: 30px;
        }
        
        
    </style>
    <div class="header-container">
        <span class="header-title">WEB MINING PROJECT <br> Mining knowledge from social media data during crisis events</span>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Barre latérale (menu déroulant) ---
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("This dashboard provides a general analysis and an analysis for each event type. Select the window you wish to view:", 
                            ["General Analysis", 'wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing'])
st.sidebar.markdown("""<div style="margin-bottom: 50px;"></div>""", unsafe_allow_html=True)


# --- General Analysis -------------------------------------------------------------
if menu == "General Analysis":
    # Subtitle
    st.markdown(
        """
        <div class="subtitle-box">
            <h3>📊 General Analysis</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Calcul general statistics
    nb_users_uniques = bd_clean['node_id_user'].nunique()
    nb_tweets_uniques = bd_clean['node_id_tweet'].nunique()
    nb_event_types = bd_clean['eventType_event'].nunique()
    types_event = bd_clean['eventType_event'].unique()
    nb_events = bd_clean['id_event_y'].nunique()
    events_uniques = bd_clean['id_event_y'].unique()
    avg_followers = bd_clean.groupby('node_id_user')['followers_count_user'].mean().mean()
    avg_followers = round(avg_followers)

    # Visualizaton on the dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Number of Users", value=f"{nb_users_uniques:,}")
    col2.metric(label="Total Number of Tweets", value=f"{nb_tweets_uniques:,}")
    col3.metric(label="Average Number of Followers per User", value=f"{avg_followers:,}")

    col1.metric(label="Number of Event Categories", value=f"{nb_event_types:,}")
    col2.metric(label="Number of Events", value=f"{nb_events:,}")

    st.write(f"List of Event Categories:")
    df_types_event = pd.DataFrame(types_event, columns=["Event Categories"])
    st.table(df_types_event)  

    st.write(f"List of Events:")
    df_id_event = pd.DataFrame(events_uniques, columns=["Events"])
    st.table(df_id_event)  


# --- Windows for each type of Events ---------------------------------------------------
elif menu in ['wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing']:
    # Subtitle
    st.markdown(
        f"""
        <div class="subtitle-box">
            <h3>Analysis of {menu.capitalize()}-Related Events</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Filter by Event
    event_tweets = bd_clean[bd_clean['eventType_event'] == menu]

    # Statistics for each event
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
    top_3_users = event_tweets.groupby('screen_name_user')['node_id_tweet'].nunique().nlargest(3).reset_index()
    top_3_users = top_3_users.rename(columns={
        'screen_name_user' :'User name',
        'node_id_tweet' : 'Number of Tweets'
    })
    priority_percentages = event_tweets.groupby('annotation_postPriority_tweet')['node_id_tweet'].nunique() / nb_tweets_event * 100
    top_5_hashtags = (
        event_tweets.groupby('id_hashtag_y')['occurences_hashtag']
        .sum()
        .nlargest(5)
        .reset_index()  
    )
    top_5_hashtags = top_5_hashtags.rename(columns={
        'id_hashtag_y': 'Hashtag',
        'occurences_hashtag': 'Number of occurrences'
    })
    top_5_hashtags['Number of occurrences'] = top_5_hashtags['Number of occurrences'].apply(lambda x: f"{x:,}")


    # Graph priority
    df_priority = pd.DataFrame(priority_percentages).reset_index()
    df_priority.columns = ['Catégorie', 'Valeur']

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

    # Visualizaton on the dashboard
    col1, col2, col3 = st.columns([2, 3, 2])
    col1.metric(label="Total Number of Tweets", value=f"{nb_tweets_event:,}")
    col2.metric(label="Number of Retweeted Tweets", value=f"{tweets_retweeted:,} ({pourcentage_retweet:.2f}%)")
    col3.metric(label="Total Number of Retweets", value=f"{total_retweets:,}")

    # Most Retweeted Tweetwith the text of the tweet
    st.write(f"Most Retweeted Tweet:")
    st.markdown(
        f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            background-color: #f1f4f8;
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

    col1, col2 = st.columns(2)
    col1.metric(label="Number of Users", value=f"{nb_users_event:,}")
    col2.metric(label="Average Number of Followers per User", value=f"{avg_followers_event:,}")
    col1.metric(label="Number of Liked Tweets", value=f"{tweets_liked:,} \n({pourcentage_likes:.2f}%)")
    col2.metric(label="Total Number of Likes", value=f"{total_likes:,}")

    st.write(f"Top 3 Users with the Most Tweets:")
    st.write(top_3_users)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Top 5 - Most Frequent Hashtags and Their Occurrences:")
    st.write(top_5_hashtags)


    # --- The temporal distribution of words and tweets across the event timeline ---------------------------------
    # Subtitle
    st.markdown(
            f"""
            <div class="subtitle-box">
                <h3>📈 Time Distribution of Published Tweets for the {menu.capitalize()} event</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        """
        <p style="font-size: 16px; font-style: italic;">In this subsection, you can visualize the distribution of tweet counts and word counts over time. The chart is interactive; you can hover over the curve to view additional details and zoom in by selecting a specific time range. You also have the option to select specific events from the legend that you want to visualize. Click the house icon (labeled 'Reset Axes') to return to the initial view.
</p>
        """,
        unsafe_allow_html=True
    )

    event_tweets['date'] = pd.to_datetime(event_tweets['date'])  
    event_tweets['word_counts'] = event_tweets['text_tweet'].apply(lambda x: len(x.split()))
    date_range = pd.date_range(event_tweets['date'].min(), event_tweets['date'].max(), freq='D')
    tweet_count_per_day = event_tweets.groupby(['date', 'id_event_y']).size().reset_index(name='count')
    tweet_pivot = tweet_count_per_day.pivot(index='date', columns='id_event_y', values='count').fillna(0)
    tweet_pivot = tweet_pivot.reindex(date_range, fill_value=0)

    word_count_per_day = event_tweets.groupby(['date', 'id_event_y'])['word_counts'].sum().reset_index(name='count')
    word_pivot = word_count_per_day.pivot(index='date', columns='id_event_y', values='count').fillna(0)
    word_pivot = word_pivot.reindex(date_range, fill_value=0)
    
    # Interactive graph - Number of tweet
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

    # Graph formatting
    fig.update_layout(
        title=f"Time Distribution of Published Tweets - {menu.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )
    # Interactive graph 
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

    # Graph formatting
    fig.update_layout(
        title=f"Time Distribution of Published Tweets - {menu.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Display graph 
    st.plotly_chart(fig)

    # Interactive graph - Number of word
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

    # Graph formatting
    fig_word.update_layout(
        title=f"Time Distribution of Published Word Count - {menu.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Number of Words in Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )
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

    # Graph formatting
    fig_word.update_layout(
        title=f"Time Distribution of Published Word Count - {menu.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Number of Words in Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Display 
    st.plotly_chart(fig_word)

    # --- TF-IDF ---------------------------------------------------------------------------

    # --- Interactifs parameters ---
    st.sidebar.subheader("TF-IDF Analysis Settings")
    st.sidebar.markdown("*You can adjust the TF-IDF analysis settings and see the changes in the chart.*")
    top_k = st.sidebar.slider("Number of Words to Display", min_value=5, max_value=5, value=50, step=10)
    num_clusters = st.sidebar.slider("Number of K-Means Clusters", min_value=2, max_value=5, value=5)
    st.sidebar.markdown("""<div style="margin-bottom: 10px;"></div>""", unsafe_allow_html=True)

    # Subtitle
    st.markdown(
        f"""
        <div class="subtitle-box">
            <h3>TF-IDF for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style="font-size: 16px; font-style: italic;">In this section, you can view the results of the TF-IDF analysis. This dashboard is interactive. You can adjust the settings in the sidebar under the 'TF-IDF Analysis Settings' section. You can select the Number of Words to Display and the Number of K-Means Clusters. After a few moments, the results will be displayed in the chart below.</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    tweets_cleaned = [preprocess_text(tweet) for tweet in event_tweets['text_tweet']]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(tweets_cleaned)
    vocab = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_
    top_k_indices = idf_scores.argsort()[-top_k:]  
    top_k_words = [vocab[i] for i in top_k_indices]
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, init="random", learning_rate=200)
    tfidf_2d = tsne.fit_transform(tfidf_matrix.T[top_k_indices].toarray())  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_2d)

    # Display on dashboard
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(top_k_words):
        x, y = tfidf_2d[i, 0], tfidf_2d[i, 1]
        plt.scatter(x, y, c=f"C{clusters[i]}", s=100) 
        plt.text(x + 0.1, y + 0.1, word, fontsize=12)

    plt.title(f"Word Clustering with the Highest TF-IDF Scores for the {menu.capitalize()} event")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.grid(True)

    # Display on dashboard Streamlit
    st.pyplot(plt)


    # --- Word embedding ---------------------------------------------------------------------------
    # Subtitle
    st.markdown(
        f"""
        <div class="subtitle-box">
            <h3>Word embedding for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <p style="font-size: 16px; font-style: italic;">This subsection displays the word embeddings of the 50 most frequent words for the '{menu}' event. This display is not interactive.</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    # Display picture on the dashboard
    image_path = os.path.join("pictures", f"Word_Embeddings-{menu}.png")
    st.image(image_path, caption=f"Word Embeddings - {menu.capitalize()}", use_container_width=True)

    # --- Tweet embedding ---------------------------------------------------------------------------
    # Subtitle
    st.markdown(
        f"""
        <div class="subtitle-box">
            <h3>Tweet embedding for {menu.capitalize()}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style="font-size: 16px; font-style: italic;">This subsection displays the tweet embeddings. This display is interactive; you can select the cluster you wish to view from the sidebar under the 'Tweet Embeddings' section. The clusters are represented by the 5 key words related to the tweet themes. This display is limited to a maximum of 10 tweets.</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<div style="margin-bottom: 20px;"></div>""", unsafe_allow_html=True)
    
    # Filter tweets by event
    tweet_embed_event = bd_tweet_embedding[bd_tweet_embedding['eventType'] == menu]
    available_clusters = tweet_embed_event["cluster_name"].unique().tolist()

    # --- Barre latérale : Select the cluster --------
    if available_clusters:
        st.sidebar.subheader("Tweet Embeddings")
        selected_cluster = st.sidebar.selectbox("Select a cluster:", available_clusters)
        filtered_tweets = tweet_embed_event[tweet_embed_event["cluster_name"] == selected_cluster][["text"]]
        filtered_tweets = filtered_tweets[["text"]].drop_duplicates()

        # --- Display tweets ---
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
                        background-color: #f1f4f8;
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

    
# --- Authors & Logo -----------------------------------
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
logo_path = os.path.join(os.getcwd(), 'pictures', 'tse_logo.png')
st.sidebar.image(logo_path, width=120)


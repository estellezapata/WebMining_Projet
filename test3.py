import streamlit as st

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# --- Données fictives pour l'exemple ---
# Remplace 'bd_clean' par tes données réelles
# Charger les fichiers CSV découpés
df1 = pd.read_csv("bd_clean_part1.csv")
df2 = pd.read_csv("bd_clean_part2.csv")
df3 = pd.read_csv("bd_clean_part3.csv")

# Réassembler les DataFrames
bd_clean = pd.concat([df1, df2, df3], ignore_index=True)

# Sauvegarder le fichier recomposé
bd_clean.to_csv("bd_clean.csv", index=False)

print("Fichier reconstitué : bd_clean.csv")
# bd_clean = pd.read_csv('bd_clean.csv')

# --- Configuration de la page ---
st.set_page_config(
    page_title="Tableau de Bord - Analyse des Événements",
    page_icon="📊",
    layout="wide"
)

# --- En-tête ---
st.markdown(
    """
    <style>
        .header {
            background-color: #2C3E50;
            padding: 15px;
            text-align: center;
            font-size: 24px;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
    <div class="header">📊 Tableau de Bord - Analyse des Données</div>
    """,
    unsafe_allow_html=True
)

# --- Barre latérale (menu déroulant) ---
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Choisissez un événement :", 
                            ["Statistiques Générales", 'wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing'])

# --- Statistiques Générales ---
if menu == "Statistiques Générales":
    st.subheader("📊 Statistiques Générales")

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
    col1.metric(label="Nombre d'utilisateurs uniques", value=nb_users_uniques)
    col2.metric(label="Nombre de tweets uniques", value=nb_tweets_uniques)
    col3.metric(label="Nombre moyen de followers par utilisateur", value=avg_followers)

    col1.metric(label="Nombre de types d'événements", value=nb_event_types)
    col2.metric(label="Nombre d'ID d'événements uniques", value=nb_events)

    # st.write(f"Types d'événements uniques : {types_event}")
    # st.write(", ".join(types_event))
    st.write(f"Types d'événements uniques :")
    # Création d'un DataFrame pour afficher les types d'événements dans un tableau
    df_types_event = pd.DataFrame(types_event, columns=["Types d'événements"])
    st.table(df_types_event)  # Affichage du tableau

    # st.write(f"ID d'événements uniques : {events_uniques}")
    # st.write(", ".join(events_uniques.astype(str)))
    st.write(f"Evénements :")
    # Création d'un DataFrame pour afficher les types d'événements dans un tableau
    df_id_event = pd.DataFrame(events_uniques, columns=["Evénements disponibles dans la base"])
    st.table(df_id_event)  # Affichage du tableau

    # st.write(f"Nombre moyen de followers par utilisateur unique : {avg_followers:.2f}")

# --- Statistiques pour un type d'événement spécifique ---
elif menu in ['wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing']:
    st.subheader(f"{menu.capitalize()} - Statistiques")

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
    top_3_users = event_tweets.groupby('name_user')['node_id_tweet'].nunique().nlargest(3)
    priority_percentages = event_tweets.groupby('annotation_postPriority_tweet')['node_id_tweet'].nunique() / nb_tweets_event * 100
    top_5_hashtags = event_tweets.groupby('id_hashtag_y')['occurences_hashtag'].sum().nlargest(5)

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
    fig = px.bar(df_priority, x="Catégorie", y="Valeur", title="Pourcentage de chaque valeur de 'annotation_postPriority_tweet'",
                labels={'Catégorie': 'Annotation Post Priority', 'Valeur': 'Pourcentage (%)'},
                color="Catégorie",  # Couleur différente pour chaque barre
                color_continuous_scale='Blues')  # Choix d'une palette de couleurs

    # Affichage des résultats pour chaque événement
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Nombre de tweets uniques", value=nb_tweets_event)
    col2.metric(label="Tweets retweetés", value=f"{tweets_retweeted} ({pourcentage_retweet:.2f}%)")
    col3.metric(label="Nombre total de retweets", value=total_retweets)

    # st.write(f"5. Le tweet le plus retweeté : {texte_tweet_max_retweet} avec {tweet_max_retweet['retweet_count_tweet']} retweets")
    # Affichage stylisé avec un encadré markdown
    st.subheader("Le tweet le plus retweeté")
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
    col1.metric(label="Nombre d'utilisateurs", value=nb_users_event)
    col2.metric(label="Nombre moyen de followers pour les utilisateurs", value=avg_followers_event)
    col1.metric(label="Nombre de tweets avec au moins un like", value=f"{tweets_liked} ({pourcentage_likes:.2f}%)")
    col4.metric(label="Nombre total de likes", value=total_likes)

    # st.write(f"6. Nombre de tweets avec au moins un like : {tweets_liked} ({pourcentage_likes:.2f}%)")
    st.write(f"Les 3 utilisateurs ayant publié le plus de tweets uniques :")
    st.write(top_3_users)
    # st.write(f"10. Pourcentage de chaque valeur de 'annotation_postPriority_tweet' :")
    # st.write(priority_percentages)

    # Affichage du graphique dans Streamlit
    # st.pyplot(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Les 5 hashtags les plus récurrents et leur nom :")
    st.write(top_5_hashtags)

    # st.plotly_chart(fig_distrib_tweet)
    # st.line_chart(tweet_count_per_day.set_index("Date"))
    # st.area_chart(tweet_count_per_day.set_index("Date"))
    # --- 📊 Graphique de distribution des tweets dans le temps ---
    st.subheader(f"📈 Distribution des tweets pour {menu.capitalize()}")

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
        title="Distribution des Tweets - Flood",
        xaxis_title="Date",
        yaxis_title="Nombre de Tweets",
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
        title="Distribution des Tweets - Flood",
        xaxis_title="Date",
        yaxis_title="Nombre de Tweets",
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
        title="Distribution des mots - Flood",
        xaxis_title="Date",
        yaxis_title="Nombre de mots dans les tweets",
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
        title="Distribution des mots dans les tweet - Flood",
        xaxis_title="Date",
        yaxis_title="Nombre de mots dans les tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        template="plotly_white"
    )

    # Afficher le graphique
    st.plotly_chart(fig_word)


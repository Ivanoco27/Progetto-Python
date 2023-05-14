import numpy as np
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import ast
from datetime import datetime
from shapely.geometry import Point
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency
import itertools
import numpy as np
from numpy import sin, cos, arccos, pi, round

def read_csv_file(file_path):
    """
    Legge un file CSV e restituisce un DataFrame.

    Args:
        file_path (str): il percorso del file CSV da leggere.

    Returns:
        pandas.DataFrame: il DataFrame corrispondente al file CSV.
    """
    df = pd.read_csv(file_path)
    print(df.info())
    return df 

def delete_col(df, col_elim):
    """
    Prende una lista delle colonne da eliminare e le elimina dal dataframe.

    Args:
        df (pandas.DataFrame): il dataframe di partenza.
        col_elim (list): la lista delle colonne da eliminare.

    Returns:
        pandas.DataFrame: il nuovo dataframe.
    """
    # Elimina le colonne specificate
    df_smart = df.drop(columns=col_elim)
    print(df_smart.info())
    return df_smart

def clean_text(text):
    """
    Prende un testo ed elimina eventuali Nan oppure restituisce un testo ripulito 
    da eventuali etichette HTML, caratteri di new line oltre a convertire il testo
    tutto in minuscolo
    
    Args: 
        text (str o obj): il testo di partenza
    
    Returns:
        cleantext (str): il testo ripulito
    
    """
    #pd.isna(text) controlla se il testo è NaN e, se sì, restituisce una stringa vuota
    if pd.isna(text):
        return ""
    #converte il testo in stringa, lo ripulisce e lo converte tutto in minuscolo
    else:
        text = str(text)
        clean = re.compile('<.*?>')
        cleantext = re.sub(clean, '', text)
        cleantext = re.sub('\r\n|\n', '', cleantext)
        cleantext = cleantext.lower()
        return cleantext
    
def change_type(df, type_col):
    """
    Trasforma il tipo delle colonne di un dataframe.

    Args:
        df (pandas.DataFrame): il dataframe di partenza.
        tipi_colonne (dict): il dizionario che associa a ogni colonna il nuovo tipo.

    Returns:
        pandas.DataFrame: il dataframe con le colonne trasformate.
    """
    df_smart= df.copy()

    # Trasforma il tipo delle colonne selezionate
    for c, new_type in type_col.items():
         if c in df.columns:
                if new_type == list:
                    df_smart[c] = df_smart[c].apply(ast.literal_eval)
                elif new_type == float:
                    df_smart[c] = df_smart[c].astype(str).str.extract('(\d+)', expand=False).astype(float)
                elif new_type == str:
                    df_smart[c] = df_smart[c].apply(clean_text).astype(str)
                elif new_type == bool:
                    df_smart[c] = df_smart[c].replace({'t':True, 'f': False}).astype(bool)
                else:
                    df_smart[c] = df_smart[c].astype(new_type)
    print(df_smart.info())
    return df_smart

def add_counts_col(df, cols):
    """
    Crea nuove colonne nel dataframe df che contengono il numero di caratteri o la durata delle colonne specificate in cols.

    Args:
        df (pandas.DataFrame): il dataframe di partenza.
        cols (list, str, datetime64[ns]): le colonne di cui calcolare la lunghezza o la durata.

    Returns:
        pandas.DataFrame: il dataframe con le nuove colonne.
    """
    for col in cols:
        new_col = "len_" + col
        if df[col].dtype == 'datetime64[ns]':
            today = datetime.today()
            df[new_col] = (today - df[col]).dt.days
        else:
            df[new_col] = df[col].apply(len)
    print(df.info())
    return df


def remove_outliers(df, column_name):
    """
    Crea un nuovo dataframe che è ripulito dagli outliers di una determinata colonna

    Args:
        df (pandas.DataFrame): il dataframe di partenza.
        column_name (float): la colonna di cui si vogliono calcolare gli outlier

    Returns:
        pandas.DataFrame: il dataframe ripulito dagli outlier
    """
    #Calcolo primo e terzo quartile e la loro differenza
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    #Calcolo lower e upper bound
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    #Creo il nuovo df privo degli outliers
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return df

def split_df(df, success_criteria):
    """
    Filtra il dataframe sulla base di criteri specificati e crea due nuovi dataframe, il primo dataframe rispetta
    i criteri e l'altro è il suo complementare.

    Args:
        df (pandas.DataFrame): il dataframe originale.
        success_criteria (dict): un dizionario contenente come chiave il nome della colonna e come valore una tupla che
        contiene il nome dell'operatore di confronto e il valore con cui confrontare.

    Returns:
        pandas.DataFrame: il dataframe che rispetta i criteri
        pandas.DataFrame: il dataframe che non rispetta i criteri
    """
    # Crea la colonna "success"
    success = []
    for idx, row in df.iterrows():
        check = True
        for col, (op, val) in success_criteria.items():
            if op == ">":
                if not row[col] > val:
                    check = False
                    break
            elif op == ">=":
                if not row[col] >= val:
                    check = False
                    break
            elif op == "<":
                if not row[col] < val:
                    check = False
                    break
            elif op == "<=":
                if not row[col] <= val:
                    check = False
                    break
            elif op == "==":
                if not row[col] == val:
                    check = False
                    break
            elif op == "!=":
                if not row[col] != val:
                    check = False
                    break
        success.append(check)

    df["success"] = success

    # Divide il dataframe in due
    success_df = df[df["success"]]
    not_success_df = df[~df["success"]]

    return success_df, not_success_df

def find_significant(df1, df2, p_value = 0.05):#era compare mean

    """
    Alla luce di 2 dataframe contenenti colonne con il medesimo nome, la funzione applica il t-test o il test del chi quadrato per tutte quelle colonne di tipo float, bool o category

    Args:
        df1 (pandas.DataFrame): il primo dataframe.
        df2 (pandas.DataFrame): il secondo dataframe.
        p_value (float): di default pari a 0.05
        
    Returns:
        significant_df (pandas.DataFrame): il dataframe che riporta le colonne dei dataframe df1 e df2 in cui le variabili sono significativamente diverse
    """
    #Creo tre liste in base al tipo della colonna e un'altra lista vuota che raccoglierà i risultati finali
    numerical_cols = [col for col in df1.columns if df1[col].dtype in ['float64', 'int64']]
    boolean_cols = [col for col in df1.columns if df1[col].dtype == 'bool']
    category_cols = [col for col in df1.columns if df1[col].dtype == 'category']
    significant_cols = []

    for col in numerical_cols:
        if col in df2.columns:
            # Calcola media e deviazione standard della colonna nei due dataframe
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            std1 = df1[col].std()
            std2 = df2[col].std()
            
             #Inizializzo delle variabili che verranno utilizzate negli altri casi
            freq1 = None
            freq2 = None
            df1_best = None
            df2_best = None
            
            # Esegue il t-test
            t, p = stats.ttest_ind(df1[col], df2[col], equal_var=False)
            
            # Verifica se la differenza tra le medie è significativa e se ciò è vero la integra nel risultato
            if p < p_value:
                significant_cols.append((col, p, mean1, mean2, freq1, freq2, df1_best, df2_best))
    
    for col in boolean_cols:
        if col in df2.columns:
            # Conta le ricorrenze di true e false nel primo dataframe
            true1 = df1[col].sum()
            false1 = len(df1) - true1
            
            # Conta le ricorrenze di true e false nel secondo dataframe
            true2 = df2[col].sum()
            false2 = len(df2) - true2
            
            #Calcolo le frequenze relative
            freq1 = true1/len(df1)
            freq2 = true2/len(df2)
            
            #Inizializzo delle variabili che verranno utilizzate negli altri casi
            df1_best = None
            df2_best = None
            mean1= None
            mean2 = None
            # Crea la tabella di contingenza
            contingency_table = pd.DataFrame({
                "DF1": [true1, false1],
                "DF2": [true2, false2]
            })
            
            # Esegui il test del chi-quadro per l'indipendenza
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Verifica se p è significativo e se ciò è vero la integra nel risultato
            if p < p_value:
                significant_cols.append((col, p, mean1, mean2, freq1, freq2, df1_best, df2_best))
    
    for col in category_cols:
        if col in df2.columns:
            
            # Conta i valori per il primo dataframe 
            df1_counts = df1[col].value_counts()
            
            # Conta i valori per il secondo dataframe            
            df2_counts = df2[col].value_counts()
            
            # Crea la tabella di contingenza            
            contingency_table = pd.DataFrame({
                "DF1": df1_counts,
                "DF2": df2_counts})
            
            # Calcolo del tipo più comune e frequenza relativa
            df1_best = df1_counts.index[0]
            df2_best = df2_counts.index[0]
            
            freq1 = df1_counts[0]/len(df1)
            freq2 = df2_counts[0]/len(df2)

            #Inizializzo le variabili che verranno utizzate in altri casi
            mean1 = None
            mean2 = None

            # Esegui il test del chi-quadro per l'indipendenza
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Verifica se p è significativo e se ciò è vero la integra nel risultato
            if p < p_value:
                significant_cols.append((col, p, mean1, mean2, freq1, freq2, df1_best, df2_best))
    
    #Trasforma la lista in un dataframe
    significant_df = pd.DataFrame(significant_cols,
                                  columns=["Column", "P-Value", "DF1 Mean", "DF2 Mean","DF1 Frequency", 
                                           "DF2 Frequency", "DF1 Best", "DF2 Best"])
    
    return significant_df 

def golden_word(df1, df2, col, p_value = 0.05):#era 

    """
    Alla luce di 2 dataframe la funzione analizza una colonna presente nei due per vedere se ci sono parole (nel caso delle stringhe) o componenti di una lista (nel caso di una lista) che abbiano una frequenza significativamente differente tra i due dataframe

    Args:
        df1 (pandas.DataFrame): il primo dataframe.
        df2 (pandas.DataFrame): il secondo dataframe.
        col (str o list): colonna oggetto dell'indagine
        p_value (float): di default pari a 0.05
        
    Returns:
        df (pandas.DataFrame): il dataframe che raccoglie tutte le parole o le componenti di una lista con una frequenza significativamente differente e con un rapporto delle frequenze tra il primo e il secondo dataframe maggiore di 1,5, il relativo p_value e il rapporto suddetto.
    """
    stop_words = set(stopwords.words('italian') + stopwords.words('english') + stopwords.words('french') + stopwords.words('german'))
    
    # Calcola la frequenza relativa delle parole nella colonna dei due dataframe
    if isinstance(df1[col].iloc[0], list):  # la colonna contiene liste di stringhe
        df1_words_counts = Counter([word for lst in df1[col] for word in lst])
        df1_total_words = sum(df1_words_counts.values())
        df1_words_freq = {word: count / df1_total_words for word, count in df1_words_counts.items()}

        df2_words_counts = Counter([word for lst in df2[col] for word in lst])
        df2_total_words = sum(df2_words_counts.values())
        df2_words_freq = {word: count / df2_total_words for word, count in df2_words_counts.items()}

    else:  # la colonna contiene stringhe normali
        df1_text = " ".join(df1[col].astype(str))
        df1_text = re.sub(r'[^\w\s]', '', df1_text)
        df1_clean_text = ' '.join([word for word in df1_text.split() if word not in stop_words])
        df1_words = df1_clean_text.split()
        df1_words_counts = Counter(df1_words)
        df1_total_words = sum(df1_words_counts.values())
        df1_words_freq = {word: count / df1_total_words for word, count in df1_words_counts.items()}

        df2_text = " ".join(df2[col].astype(str))
        df2_text = re.sub(r'[^\w\s]', '', df2_text)
        df2_clean_text = ' '.join([word for word in df2_text.split() if word not in stop_words])
        df2_words = df2_clean_text.split()
        df2_words_counts = Counter(df2_words)
        df2_total_words = sum(df2_words_counts.values())
        df2_words_freq = {word: count / df2_total_words for word, count in df2_words_counts.items()}

    # Esegue il test del chi-quadrato per ogni parola significativa
    
    common_words = set(df1_words_freq.keys()) & set(df2_words_freq.keys())
    result = []
    
    for word in common_words:
        obs = [[df1_words_counts[word], df2_words_counts[word]],
                [df1_total_words - df1_words_counts[word], df2_total_words - df2_words_counts[word]]]
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        freq_ratio = df1_words_freq[word] / df2_words_freq[word]
        
        # Filtro per parole significative
        if p < p_value and freq_ratio > 1.5:
            result.append((word, p, freq_ratio))

    result = sorted(result, key=lambda x: x[1])
    df = pd.DataFrame(result, columns=['word', 'p_value', 'freq_ratio'])

    return df

#insieme di funzioni per trovare distanza date 2 coordinate terrestri
#si basa sulla formula di Haversine, a differenza della distanza euclidea,
#tiene conto della curvatura terrestre attraverso la trigonometria

#restiuisce, date due coordinate, la distanza in metri approssimata a due decimali 

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians

def distanza(latitude1, longitude1, latitude2, longitude2):
    theta = longitude1 - longitude2
    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) + 
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        ))
    return round(distance * 1609.344, 2) 

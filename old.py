mport multiprocess as mp
import time
def parallelize_extraction(texts_array):
    chunks = np.array_split(range(len(texts_array)), cpu_count())
    def extract_main_lang_worker(chunk):
        textlist= []
        for t in texts_array[chunk]:
            sentences = artificial_sentence_split(t)
            main_lang, lang_dict = text_lang_split_2(t)
            text_to_keep = lang_dict[main_lang]["text"]
            textlist.append(text_to_keep)
        return textlist
    pool = mp.Pool(mp.cpu_count())

    results = pool.map(extract_main_lang_worker, (chunks))
    pool.close()
    return [r[0] for r in results]

texts_array = df["texts"].values
a = parallelize_extraction(texts_array)

#### OLD

def get_row_for_df_v2(json_line): #obsolete
    """ extract items from json and returns a row to put in a dataframe """
    code = json_line['code']
    if "ocrs" in json_line:
        langs = []
        confidences = []
        texts = []
        keys =  list(json_line['ocrs'].keys())
        for key in keys:
            ocr_text = json_line['ocrs'][key]['text']
            #detected_langs = json_line['ocrs'][key]['detectedLanguages']
            main_lang, sorted_dict = text_lang_split(ocr_text)
            text_main_lang = sorted_dict[main_lang]["text"]
            confidence = np.mean(sorted_dict[main_lang]["prob"])
 
            #main_lang, confidence = get_main_lang(detected_langs)
            texts.append(text_main_lang)
            langs.append(main_lang)
            confidences.append(confidence)
    row = [code, "<end_of_text> \n".join(texts),  confidences, langs, keys]
    return row





    def get_row_for_df(json_line): # obsolete
    """ extract items from json and returns a row to put in a dataframe """
    code = json_line['code']
    if "ocrs" in json_line:
        langs = []
        confidences = []
        texts = []
        keys =  list(json_line['ocrs'].keys())
        for key in keys:
            ocr_text = json_line['ocrs'][key]['text']
            detected_langs = json_line['ocrs'][key]['detectedLanguages']
            main_lang, confidence = get_main_lang(detected_langs)
            texts.append(ocr_text)
            langs.append(main_lang)
            confidences.append(confidence)
    row = [code, "<end_of_text> \n".join(texts),  confidences, langs, keys]
    return row


    def get_main_lang(dict_list): #obsolete
    """ get the language that has the highest confidence in 'detectedLanguages' """
    max_v = 0
    main_lang = "not_found"
    for dict_ in dict_list:
        lang = dict_["languageCode"]
        if "confidence" in dict_:
            confidence = dict_["confidence"]
        else: confidence = -1
        if confidence > max_v:
            max_v = confidence
            main_lang = lang
    return main_lang, max_v



  def get_lang(text, threshold = 0.6): #obsolete
    """takes text as input and returns main language detected if its probability is above the threshold"""
    main_lang = 'not_found'
    max_prob = 0
    try :
        langs = detect_langs(text)
        for lang in langs:
            if lang.prob > max(max_prob, threshold): 
                max_prob = lang.prob 
                main_lang = lang.lang
    except:
        main_lang = "not_found"
        max_prob = 0

    return main_lang, max_prob



def text_lang_split(text:str): #obsolete
    """
    takes text as input and splits it in a dictionnary with languages as keys.
    for each language we have subkeys such as:
    text: text found with the given language
    len_text: the length of the text
    prob: a list of probabilities, each probability corresponds to a sentence.  
    """
    text = re.sub(r"\n", " ", text)
    text = re.sub("\.+", ".", text)
    lang_dict = {}
    for sentence in artificial_sentence_split(text):
        #lang, prob = get_lang(sentence)
        lang, prob = get_lang_2(sentence)
        if lang in lang_dict:
            lang_dict[lang]["prob"].append(prob)
            lang_dict[lang]["len_text"] += len(sentence)
            lang_dict[lang]["text"]+= sentence
            
        else:
            lang_dict[lang] = {}
            lang_dict[lang]["prob"] = [prob]
            lang_dict[lang]["len_text"] = len(sentence)
            lang_dict[lang]["text"] = sentence
            
    sorted_dict = {k: v for k, v in sorted(lang_dict.items(), key=lambda item: item[1]["len_text"], reverse = True)}
    main_lang = next(iter(sorted_dict))
    return main_lang, sorted_dict



 def get_row_for_df_lite(json_line):
    """ extract items from json and returns a row to put in a dataframe """
    code = json_line['code']
    if "ocrs" in json_line:
        texts = []
        keys =  list(json_line['ocrs'].keys())
        for key in keys:
            ocr_text = json_line['ocrs'][key]['text']
            texts.append(ocr_text)
    row = [code, "<end_of_text> \n".join(texts), keys]
    return row




def floor(x:int)->int:
    if x >= 1000:
        x = 1000
    return x
def make_score(confidences):
    if len(confidences)==1:
        malus = 0.7
    else:
        malus = 1
    return np.mean(confidences)*malus

def get_most_confident_lang(confidences, langs, texts): #obsolete
    """
    get the language that has a good balance between confidence and length
    score = confidence_mean * text_len. if text_len > 600 then text_len = 600
    """
    D = {}
    for conf, lang, text in zip(confidences, langs, texts.split("<end_of_text> \n")):
        if lang in D:
            D[lang]["confidences"].append(conf)
            D[lang]["text_len"] += len(text)
        else:
            D[lang] = {}
            D[lang]["confidences"] = [conf]
            D[lang]["text_len"] = len(text)
    output_dict = {lang: make_score(D[lang]['confidences'])*floor(D[lang]['text_len']) for lang in D}
    lang_to_keep = max(output_dict, key = output_dict.get)
    lang_to_keep_confidence = np.mean(D[lang]["confidences"])
    return lang_to_keep, lang_to_keep_confidence, D, output_dict

def extract_text_with_lang(langs, lang_to_keep, text, split_key = "<end_of_text> \n"):
    to_pick = np.where(np.array(langs) ==lang_to_keep)[0]
    text_as_a_list = text.split(split_key)
    output_text = " ".join([text_as_a_list[i] for i in to_pick])
    return output_text

def extract_text_with_lang_from_df(df):
    extracted_texts = []
    lang_to_keep_list = []
    lang_to_keep_confidence_list = []
    for confidences, langs, texts in zip(df["confidences"], df["langs"], df["texts"]):
        lang_to_keep, lang_to_keep_confidence, _, _ = get_most_confident_lang(confidences, langs, texts)
        output_text = extract_text_with_lang(langs, lang_to_keep, texts, split_key = "<end_of_text> \n")
        #append items
        extracted_texts.append(output_text)
        lang_to_keep_list.append(lang_to_keep)
        lang_to_keep_confidence_list.append(lang_to_keep_confidence)
    return extracted_texts, lang_to_keep_list, lang_to_keep_confidence_list

from tqdm import tqdm
freq_dict = {}
for text in tqdm(df_l["text_cleaned"]):
    for word in remove_duplicates(text).split():
        if str(word).lower() in freq_dict:
            freq_dict[str(word).lower()] += 1
        else:
            freq_dict[str(word).lower()] =

freq_dic_sorted = {k: v for k, v in sorted(freq_dict.items(), reverse = True, key=lambda item: item[1]) if v > 1}



"TFIDF TRICK___________"


def get_index_to_word_dict(vectorizer):
    """ make the dictionnary index_to_word
    Parameters
    ----------
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
        tfidf_vectorizer instance fit on a sub_dataframe. 
    
    Returns
    -------
    index_to_word: dict
        maps a word index (vectorizer.vocabulary_) to its associated word.
    """
    index_to_word = {index: word for index, word in zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys())}
    return index_to_word

def get_words_with_highest_scores(scores, words, n_words_to_take):
    """
    takes scores and words of an OCR as inputs and returns 
    words with highest tfidf scores

    Parameters
    ----------
    scores: list
        list of tfidf scores.
    words: list
        list of words
    n_words_to_take: int
        number of words to select.

    Returns
    -------
    best_items: list
        list of words with highest scores.
    items: list:
        all items
    """
    items = [item for item in sorted(zip(scores, words), reverse = True)]
    best_items = items[:n_words_to_take]
    return best_items, items


def get_words_and_scores_from_tfidf_matrix(doc, cols, tfidf_matrix, index_to_word):
    """takes a document (line of tfidf matrix) and its words (columns of tfidf matrix
    and returns words of the documents with their ifidf scores
    columns are the non zero values of the tfidf matrix.

    Parameters
    ----------
    doc: int
        document index.
    cols: int
        column indexes.
    tfidf_matrix: scipy.sparse.csr.csr_matrix
        tdidf sparse matrix.
    index_to_word: dict
        maps a word index (vectorizer.vocabulary_) to its associated word.

    Returns
    -------
    best_items: list
        list of words with highest scores.
    items: list:
        list of all words.
    """
    scores = []
    words = []
    for col in cols:
        word = index_to_word[col] 
        score = tfidf_matrix[doc, col]
        words.append(word)
        scores.append(score)
    return words, scores

def text_selection(df_idx, sub_df_idx,  index_to_word, tfidf_matrix, n_words_to_take = 50):
    """ makes the text selection on a document based on the tfidf score
    
    Parameters
    ----------
    df_idx: int
        index of the document in the original dataframe
    sub_df_idx: int
        index of the document in the sub-dataframe (filtered dataframe on a given language)
    index_to_word: dict
        maps a word index (vectorizer.vocabulary_) to its associated word.
    tdidf_matrix: scipy.sparse.csr.csr_matrix
        tdidf sparse matrix for a given language.
    n_words_to_take: int
        total words to keep.
    
    Returns
    -------
    text_selection: str
        text selected after the tfidf trick.
    items: list
        list where each element is a tuple in the following format (tdidf_score, word).
        All the words in the original text are kept.
    """
    rows, cols = tfidf_matrix[sub_df_idx].nonzero()
    ## extract words and scores from tfidf matrix
    words, scores = get_words_and_scores_from_tfidf_matrix(sub_df_idx, cols, tfidf_matrix, index_to_word)
    #extract words with highest score from sentence
    best_items, items = get_words_with_highest_scores(scores, words, n_words_to_take)
    best_words = [item[1] for item in best_items]
    
    text_selection = " ".join([word for word in df["text_cleaned"].iloc[df_idx].split() if str(word).lower() in best_words])
    text_selection = " ".join(best_words)
    #text_selection_unique = remove_duplicates(text_selection)
    return text_selection, items

def text_selection_from_Series(text_Series, tfidf_matrix, index_to_word):
    """ takes a text pd.Series as input and makes the tfidf-selection for each document

    Parameters
    ----------
    text_Series: pd.Series
        pd.Series containing OCR text
    tdidf_matrix: scipy.sparse.csr.csr_matrix
        tdidf sparse matrix for a given language.
    index_to_word: dict
        maps a word index (vectorizer.vocabulary_) to its associated word.

    Returns
    -------
    text_selection_list: list
        list containing all the texts after the tfidf-selection
    items_list: list
        each element of items_list is a list containing items.
        An item is in the following format (tdidf_score, word)
        All the words in the original text are kept.
    """
    text_selection_list = []
    items_list = []
    for sub_df_idx, df_idx in enumerate(tqdm(text_Series.index)):
        text_selection_unique, items = text_selection(df_idx, sub_df_idx, index_to_word, tfidf_matrix, n_words_to_take = 30)
        text_selection_list.append(text_selection_unique)
        items_list.append(items)
    return text_selection_list, items_list

"""takes approx 6min"""
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import clear_output
problematic_langs = []
df = df.reset_index(drop = True)
for lang in df["main_lang"].unique():
    print("lang:", lang)
    lang_filter = df["main_lang"]== lang 
    sub_df = df[lang_filter]
    try:
        vectorizer = TfidfVectorizer(min_df = 1, max_df = 0.8)
        tfidf_matrix = vectorizer.fit_transform(sub_df["text_cleaned"])
        index_to_word = get_index_to_word_dict(vectorizer)
        text_selection_list, items_list =  text_selection_from_Series(sub_df["text_cleaned"], tfidf_matrix, index_to_word)
        df.loc[lang_filter, "tfidf_selection"] = text_selection_list
    except:
        problematic_langs.append(lang)
    clear_output()




"""
remarque OCR en russe probleme marche pas tres bien
avoir les txtannotations pour les produits

"""



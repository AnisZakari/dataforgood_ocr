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
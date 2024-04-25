import torch
import statistics
import csv
import numpy as np
from scipy import stats
from torch.nn.functional import cosine_similarity as torch_cosine
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import cltk
import requests
import json
import xml.etree.ElementTree as ET
import re
import string
import spacy

spacy_nlp = spacy.load('grc_proiel_trf')

cltk_nlp = cltk.NLP(language="grc")

stop = "μή, δ, ἑαυτοῦ, ἄν, ἀλλ', ἀλλά, ἄλλος, ἀπό, ἄρα, αὐτός, δ', δέ, δή, διά, δαί, δαίς, ἔτι, ἐγώ, ἐκ, ἐμός, ἐν, ἐπί, εἰ, εἰς, γάρ, γε, γα, ἡ, ἤ, καί, κατά, μέν, μετά, μή, ὁ, ὅδε, ὅς, ὅστις, ὅτι, οὕτως, οὗτος, οὔτε, οὖν, οὐδείς, οἱ, οὐ, οὐδέ, οὐκ, περί, πρός, σύ, σύν, τά, τε, τήν, τῆς, τῇ, τι, τί, τις, τίς, τό, τοί, τοιοῦτος, τόν, τούς, τοῦ, τῶν, τῷ, ὑμός, ὑπέρ, ὑπό, ὡς, ὦ, ὥστε, ἐάν, παρά, σός"
stop_words = [word for word in stop.split(", ")]
stop_words.append('̓')
stop_words.append(",")

with open("tlg0012.tlg001.perseus-grc2.json", "r", encoding = "utf-8") as file_1:
    corpus_1 = json.load(file_1)

with open("tlg0012.tlg002.perseus-grc2.json", "r", encoding = "utf-8") as file_2:
    corpus_2 = json.load(file_2)

corpora = [corpus_1, corpus_2]

def remove_punctuation(input_string):
    return re.sub(f"[{re.escape(string.punctuation)}]", "", input_string)

def remove_diacritics(text_list):
    import unicodedata
    return [''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn') for text in text_list]


def mean_cls_embeddings(phrases_list, model, tokenizer):

    """
    Function that, given a list of phrases in ancient greek,a model and a tokenizer
    returns a dictionary that contains two dictionaries: one (cls_emb_dict) with the 
    cls embedding for each phrase, taken from the last layer of the specified model, 
    the other (mean emb_dict) with the mean of the embeddings of each word for each phrase,
    taken also from the last layer.
    """

    input_dict = {}
    output_dict = {}
    emb_dict = {}
    emb_dict["cls_emb_dict"] = {}
    emb_dict["mean_emb_dict"] = {}

    for i, phrase in enumerate(phrases_list):
        input_dict[f"encoded_input_{i}"] = tokenizer(phrase, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output_dict[f"model_output_{i}"] = model(**input_dict[f"encoded_input_{i}"]) 
        emb_dict["cls_emb_dict"][f"cls_embedding_{i}"] = output_dict[f"model_output_{i}"].hidden_states[-1][0][0].unsqueeze(0)                      
        emb_dict["mean_emb_dict"][f"mean_embedding_{i}"] = output_dict[f"model_output_{i}"].hidden_states[-1].mean(dim=1)

    return emb_dict


def report_similarity(original_emb_dict, paraphrased_emb_dict, confound_emb_dict, method = "", report = True):

    """
    Function that takes as input the dictionaries of embeddings for the original 
    sentences, the paraphrased and the confounds. It then creates a dictionary where
    are stored the average and the variance of the similarity scores between original 
    and paraphrased and original and confound versions of the same sentences, and also 
    the same metrics but for the similarity scores between two original sentences (to 
    act as a baseline that should be as low as possible). 

    If the parameter report is True, when called, the function also prints the t test
    between: 
    - the original-paraphrased similarity score distribution and the original-confound (how
    much the scores' distribution is different (t statistic) and if the difference is 
    statistically relevant (p-value))
    - the original-paraphrased similarity score distribution and the original[i]-original[i+1] 
    (how much the scores' distribution is different (t statistic) and if the difference is 
    statistically relevant (p-value))
    """

    similarity_scores = {"originalvsparaphrased": {},
                         "originalvsconfound": {},
                         "originalvsoriginal": {},
                         }
    o_p_similarity = []
    o_c_similarity = []
    o_o_similarity = []

    for i in range(len(original_emb_dict[f"{method}_emb_dict"])):
        o_p_similarity.append(torch_cosine(original_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{i}"], paraphrased_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{i}"]))
    
    for i in range(len(original_emb_dict[f"{method}_emb_dict"])):
        o_c_similarity.append(torch_cosine(original_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{i}"], confound_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{i}"]))

    count = 1
    for i in range(len(original_emb_dict[f"{method}_emb_dict"])):
        if count > (len(original_emb_dict[f"{method}_emb_dict"]) - 1):
            count = 0
        o_o_similarity.append(torch_cosine(original_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{i}"], original_emb_dict[f"{method}_emb_dict"][f"{method}_embedding_{count}"]))
        count += 1

    similarity_scores["originalvsparaphrased"]["average"] = np.mean(o_p_similarity)
    similarity_scores["originalvsparaphrased"]["variance"] = np.var(o_p_similarity)

    similarity_scores["originalvsconfound"]["average"] = np.mean(o_c_similarity)
    similarity_scores["originalvsconfound"]["variance"] = np.var(o_c_similarity)

    similarity_scores["originalvsoriginal"]["average"] = np.mean(o_o_similarity)
    similarity_scores["originalvsoriginal"]["variance"] = np.var(o_o_similarity)

    if report:
        para_t_stat, para_p_value = stats.ttest_ind(o_p_similarity, o_o_similarity)
        print(f"T-statistic between original-paraphrased and original-original: {para_t_stat}, P-value: {para_p_value}")

        conf_t_stat, conf_p_value = stats.ttest_ind(o_c_similarity, o_o_similarity)
        print(f"T-statistic between original-confound and original_original: {conf_t_stat}, P-value: {conf_p_value}")

        pc_t_stat, pc_p_value = stats.ttest_ind(o_p_similarity, o_c_similarity)
        print(f"T-statistic between original-paraphrased and original-confound: {pc_t_stat}, P-value: {pc_p_value}")

    return similarity_scores


def create_wordnet(path):
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        wordnet = {}

        for row in reader:
            if not row[0].startswith("#"):
                synset = row[0]
                lemma = row[2]
                if synset not in wordnet.keys():
                    wordnet[f"{synset}"] = []
                wordnet[f"{synset}"].append(lemma)

    return wordnet


def lemmatize_grecy(string):
    doc = spacy_nlp(string)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def lemmatize_cltk(string):
    doc = cltk_nlp.analyze(text=string)
    lemmas = [lemma for lemma in doc.lemmata]
    return lemmas


def pos_tag_grecy(string):
    tag_mapping = {
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a'
    }
    doc = spacy_nlp(string)
    pos = [token.pos_ for token in doc]
    pos_tag = [tag_mapping.get(tag, tag) for tag in pos]
    return pos_tag


def pos_tag_cltk(string):
    tag_mapping = {
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a'
    }
    doc = cltk_nlp.analyze(text=string)
    pos = doc.pos
    pos_tag = [tag_mapping.get(tag, tag) for tag in pos]
    return pos_tag


def get_list_lemmata():

    lemmata_list = []
    
    for corpus in corpora:

        for sent_id in range(1, len(corpus)+1):

            for word_dic in corpus[f"{sent_id}"]:

                for lemma_pos in word_dic[f"lemmas_pos"]:
                    lemmata_list.append(lemma_pos[0])

    return lemmata_list


def lemmatize_corpus(corpus, sent_id):
    lemmata = []

    for word_dic in corpus[f"{sent_id}"]:
        word_lemmata = []

        for lemma_pos in word_dic[f"lemmas_pos"]:
            word_lemmata.append(lemma_pos[0])
        
        lemmata.append(word_lemmata)

    return lemmata

def pos_tag_corpus(corpus, sent_id):
    pos_tags = []

    for word_dic in corpus[f"{sent_id}"]:
        word_pos = []

        for lemma_pos in word_dic[f"lemmas_pos"]:
            word_pos.append(lemma_pos[1])
        
        pos_tags.append(word_pos)

    return pos_tags


def get_inflected_form(lemma, corpus, sent_id):

    for i in range(1, len(corpus)+1):

        if i == sent_id:
            for word_dic in corpus[str(sent_id)]:

                for lemma_pos in word_dic[f"lemmas_pos"]:
                    if lemma_pos[0] == lemma:
                        word = word_dic["word_form"]
                        break
                
    return word


def greeklemma2offset(lemma, pos):
    offsets = []

    for offset in wordnet:
        offset_pos = offset.split("-")[1]

        if lemma in wordnet[f"{offset}"] and offset_pos == pos:
            offsets.append(offset)
    return offsets


def offset2greeklemma(offsets, original_word = ""):
    lemmata = []
    for offset in offsets:

            for lemma in wordnet[offset]:
                if lemma not in lemmata and lemma in subcorpus_lemmata_list and lemma != original_word:
                    lemmata.append(lemma)

    return lemmata


def get_synonims(original, lemmatization = "grecy"):

    """
    Function that takes as input an original sentence and a method for lemmatization (grecy or cltk) and, using the Ancient
    Greek Wordnet, returns a dictionary of the following structure:

    
    {"lemma":   
        (pos, [list of synonims for that particular lemma])      
        }
        
    """

    if lemmatization == "cltk":
        lemmata_original = lemmatize_cltk(original)
        pos_original = pos_tag_cltk(original) 
    elif lemmatization == "grecy":
        lemmata_original = lemmatize_grecy(original)
        pos_original = pos_tag_grecy(original)

    print(lemmata_original)
    
    synonyms = {}
        
    for idx, lemma in enumerate(lemmata_original):

        if lemma not in stop_words: # we don't compute the synonims for the stopwords, since they have no semantic importance 
            pos = pos_original[idx]

            # try:
            offsets = greeklemma2offset(lemma, pos)

            syn_list = offset2greeklemma(offsets, original_word = lemma)
            
            if syn_list != []:
                synonyms[lemma] = (pos, syn_list)

            # try:
            #     lemma_r = requests.get(f"https://greekwordnet.chs.harvard.edu/api/lemmas/{lemma}/synsets", verify=False)
            #     lemma_dic = lemma_r.json()

            #     if lemma_dic["results"]:
            #         for i in len(lemma_dic["results"]):
                        
            #             if lemma_dic["results"][i]["pos"] == pos and lemma_dic["results"][i]["synsets"]["literal"]:
            #                 synsets_pos = [(dic["offset"], dic["pos"]) for dic in lemma_dic["results"][0]["synsets"]["literal"]]
                    
            #         for synset, pos in synsets_pos:
            #             # synonims[f"sentence {i}"][lemma][f"{sense}"] = []
            #             synsets_r = requests.get(f"https://greekwordnet.chs.harvard.edu/api/synsets/{pos}/{synset}/lemmas", verify=False)
            #             synsets_dic = synsets_r.json()
            #             synonims[f"sentence {i}"][lemma][f"{synset}"] = (pos,[sin_dict["lemma"] for sin_dict in synsets_dic["results"][0]["lemmas"]["literal"]])

            # except Exception as e:
            #     print(f"Error processing lemma {lemma}: {e}")
            #     # Optionally, handle specific exceptions more granularly
            
    return synonyms


def find_sentence_with_lemma(pos, lemma):
    sents = []
    for corpus_id, corpus in enumerate(corpora):

        for sent_id in range(1, len(corpus)+1):

            for word_dic in corpus[f"{sent_id}"]:

                for lemma_pos in word_dic[f"lemmas_pos"]:
                    corpus_lemma = lemma_pos[0]
                    corpus_pos = lemma_pos[1]
                
                    if corpus_lemma is not None and lemma == corpus_lemma and pos == corpus_pos :
                        print(lemma)
                        word_forms = [dic["word_form"] for dic in corpus[f"{sent_id}"]]
                        sent = " ".join(word_forms) 
                        lemma_id = f"{corpus_id}_{sent_id}"
                        id_sents = (lemma_id, sent)
                        sents.append(id_sents)
                        break
            
            if len(sents) == 4:
                break
    
    return sents

def get_w2v_embedding(word, sentence):
    doc = cltk_nlp.analyze(sentence)
    word_id = doc.tokens.index(word)
    word_emb = doc.embeddings[word_id]
    return word_emb 

def get_word_embedding(word, model, tokenizer):

    """
    Function that, given a word, a BERT model and a tokenizer, extracts
    the BERT embedding of the word from the last layer by computing the
    mean over the embeddings of the possible subwords.
    """

    tokens = ['[CLS]'] + tokenizer.tokenize(word) + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]) # converting token ids into tensor with batch dimension 

    with torch.no_grad():
        outputs = model(input_tensor)
        hidden_states = outputs.hidden_states[-1]  # shape: (1, number of tokens, 768)
    
    word_embedding = hidden_states[0, 1:-1, :].mean(dim=0) 
    return word_embedding


def get_word_in_sent_embedding(sentence, model, tokenizer):

    """
    Function that, given a sentence, a model and a tokenizer, 
    computes the embedding of each word in the sentence as the mean
    over the embeddings of its subwords. Returns a list of embeddings,
    one for each word
    """

    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states[-1]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    word_tokens_map = []
    for token in tokens:
        if token.startswith("##"):
            word_tokens_map[-1].append(token)
        else:
            word_tokens_map.append([token])

    word_embeddings = []

    for word_tokens in word_tokens_map:
        # get the indices of the tokens in the original list of tokens
        token_indices = [tokens.index(token) for token in word_tokens]
        # aggregate the embeddings of the tokens for each word
        word_embedding = torch.mean(hidden_states[0, token_indices, :], dim=0)
        word_embeddings.append(word_embedding)

    return word_embeddings

def w2v_sentence2syn_embeddings(synonym, id_syn, synonym_sent):
    corpus_id = int(id_syn.split("_")[0]) #id of the corpus in which the synonym  is present
    sent_id = int(id_syn.split("_")[1]) # id of the sentence in which the synonym is present
    inflected_synonym = get_inflected_form(synonym, corpora[corpus_id], sent_id)
    syn_embedding = get_w2v_embedding(inflected_synonym, synonym_sent) # we get the w2v embedding for the synonym
    return syn_embedding
            

def bert_sentence2syn_embedding(synonym, id_syn, syn_sent, model, tokenizer):
    corpus_id = int(id_syn.split("_")[0]) #id of the corpus in which the synonym  is present
    sent_id = int(id_syn.split("_")[1]) # id of the sentence in which the synonym is present
    syn_sentence_embeddings = get_word_in_sent_embedding(syn_sent, model, tokenizer)
    syn_lemmata = lemmatize_corpus(corpora[corpus_id], sent_id) # list of lemmatizations per word in sentence (i need it to
    # get the correct embedding index)

    for i, list_lemmata in enumerate(syn_lemmata): # for every index, list of different lemmatizations per word in sentence containing synonym

        if synonym in list_lemmata: # when we find the right list of lemmatizations
            return syn_sentence_embeddings[i] # we append the corresponding  contextual embedding



def get_synonym_embeddings(pos_list_synonyms, w2v = True, model = None, tokenizer = None):

    """
    Function that, given a list of synonyms, a model and a tokenizer,
    computes BERT's contextual embedding or the word2vec embedding 
    of each synonym in the synset in four different sentences 
    taken from an ancient greek corpus.
    Returns a list of tuples (synonym, [embeddings]).
    """

    synonyms_embeddings_t = []

    pos = pos_list_synonyms[0]
    list_synonyms = pos_list_synonyms[1]

    for synonym in list_synonyms:
        synonym_embeddings = []
        id_sentences = find_sentence_with_lemma(pos, synonym)

        for id_syn, syn_sent in id_sentences:

            if w2v == True:
                synonym_embeddings.append(w2v_sentence2syn_embeddings(synonym, id_syn, syn_sent))
            else:
                synonym_embeddings.append(bert_sentence2syn_embedding(synonym, id_syn, syn_sent, model, tokenizer))
            
        syn_emb_tup = (synonym, synonym_embeddings)
        if syn_emb_tup[1] != []:
            synonyms_embeddings_t.append(syn_emb_tup)

    return synonyms_embeddings_t

def get_w2v_similarity(word_embedding, synonym_embedding):
    # reshape numpy arrays to use cosine similarity
    synonym_embedding = synonym_embedding.reshape(1, -1)
    word_embedding = word_embedding.reshape(1, -1)
    return sklearn_cosine(synonym_embedding, word_embedding)


def get_bert_similarity(word_embedding, synonym_embedding):
    # adjusting the dimensions of word and sense embedding:
    synonym_embedding = synonym_embedding.unsqueeze(0) if synonym_embedding.dim() == 1 else synonym_embedding 
    word_embedding = word_embedding.unsqueeze(0) if word_embedding.dim() == 1 else word_embedding
    return torch_cosine(synonym_embedding, word_embedding) 



def dic_similarities(syn_dic, sentence, w2v = True, model = None, tokenizer = None, lemmatization = "grecy"):

    """
    Function that, given a dictionary of synonims, a list of original sentences
    for which it must find synonims, a model and a tokenizer, returns a
    dictionary of the following structure:
    sents_sim = {"original sentence i": 
                    {"word i": [(sense i, similarity score i), (sense i+1, similarity score i+1)
                            ....]
                        }
                    }
    where the similarity scores are calculated between the contextual embeddings of every
    synonym in a synset (computed by extracting 4 different sentences containing the synonym from 
    a corpus) and the contextual embedding of the word: for each synonym only the highest score
    between the 4 different embeddings is registered
    """

    sents_sim = {}

    if w2v == False:
        sent_embeddings = get_word_in_sent_embedding(sentence, model, tokenizer) # getting the contextual embedding for each word in the sentence

    sentence_no_punct = remove_punctuation(sentence)

    if lemmatization == "grecy":
        lemmata = lemmatize_grecy(sentence_no_punct) # getting the lemmas of each word in the sentence

    elif lemmatization == "cltk":
        lemmata = lemmatize_cltk(sentence_no_punct) # getting the lemmas of each word in the sentence

    for lemma in syn_dic:
        sents_sim[lemma] = []
        for i, l in enumerate(lemmata):
            if l == lemma:
                word_id = i
        # word_id = lemmata.index(lemma)

        if w2v == True:
            word_embedding = get_w2v_embedding(sentence_no_punct.split(" ")[word_id], sentence) # getting the word2vec embedding for the current word for which we want a synonim
        else:
            word_embedding = sent_embeddings[word_id] # getting the contextual embedding for the current word for which we want a synonim

        synonym_embeddings = get_synonym_embeddings(syn_dic[lemma], w2v = w2v, model=model, tokenizer=tokenizer) # getting the synonyms' embeddings

        for synonym, embeddings in synonym_embeddings:
            similarities = []

            for synonym_embedding in embeddings:
                if w2v == True:
                    sim = get_w2v_similarity(word_embedding, synonym_embedding)
                else:
                    sim = get_bert_similarity(word_embedding, synonym_embedding)

                similarities.append(sim)

            similarities.sort()
            sim_tup = (synonym, similarities[-1])
            sents_sim[lemma].append(sim_tup)

    return sents_sim

def report_similarities(sim_dic):

    """
    Function that prints for each word in each original sentence its best
    synonym with the corresponding similarity score.
    """

    for lemma in sim_dic:
        synonyms = {}
        # print(synonyms)
        sorted_values = sorted(sim_dic[lemma], key=lambda x: x[1], reverse=True)
        print(f"The best synonyms for {lemma} are:")

        for syn, value in sorted_values[:20]:
            if syn in synonyms and synonyms[f"{syn}"] < value:
                    synonyms[f"{syn}"] = value
                    print(f"\t{syn} with a score of {value}")
            elif syn not in synonyms:
                synonyms[f"{syn}"] = value
                print(f"\t{syn} with a score of {value}")
            else:
                continue

wordnet = create_wordnet("wn-data-grc.tab")
subcorpus_lemmata_list = get_list_lemmata()

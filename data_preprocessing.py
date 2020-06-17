import json, os, sys, argparse, time, itertools, glob, gc, csv
import numpy as np
from fast_scripts import generate_placeholder
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from keras.preprocessing.sequence import pad_sequences
from keras.layers import  Embedding
from mytext import Tokenizer
from tests import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy import spatial
from sklearn.utils import class_weight
from constants import POLITICAL_PARTIES, DOMAIN_CLASSES, TERRITORIAL_CLASSES, SUBDOMAIN_CLASSES, \
    COMPLETE_CLASSES, MINI_SUBDOMAIN_CLASSES, EMB_FOLDER, BRITISH_MANIFESTOS_DOMAINS, BRAULIO_CLASSES, \
    SUBBRAULIO_CLASSES, BRAULIO_CLASSES_NO_REBABA, GLOBAL_MANIFESTOS_DOMAINS, GLOBAL_MANIFESTOS_SUBDOMAINS, COUNTRY_PARTIES,SUBSUB_TO_SUB,BANNED_SUBS_FOR_HT

LANGUAGES_DICT = {'german': 'de', 'danish': 'da', 'spanish': 'es', 'italian': 'it', 'english': 'en', 'finnish': 'fi',
                  'french': 'fr'}


def flatten_array(data):
    return [x for sublist in data for x in sublist]

def one_hot_encoder(total_classes):
    encoder = {}
    array_size = len(total_classes)
    for i, code in enumerate(total_classes):
        vector = np.zeros(array_size)
        vector[i] = 1.0
        encoder[code] = vector.tolist()
    return encoder

def to_one_hot_encoding(y,classes, sequential = False):
    encoder = one_hot_encoder(classes)
    new_y = []
    if not sequential:
        for label in y:
            new_y.append(encoder[label])
    else:
        for split in y:
            new_split_y = []
            for y_split in split:
                new_split_y.append(encoder[y_split])
            #print new_split_y
            new_y.append(new_split_y)
    return new_y

def find_1(one_hot):
    for i,e in enumerate(one_hot):
        if e == 1:
            return i
    print "not found"


def get_classes_from_target_class(class_objective):
    classes = None
    if class_objective == "domain":
        classes = DOMAIN_CLASSES
    elif class_objective == "territorial":
        classes = TERRITORIAL_CLASSES
    elif class_objective == "subdomain":
        classes = SUBDOMAIN_CLASSES
    elif class_objective == "complete":
        classes = COMPLETE_CLASSES
    elif class_objective == "mini_subdomain":
        classes = MINI_SUBDOMAIN_CLASSES
    elif class_objective == "party":
        classes = POLITICAL_PARTIES
    elif class_objective == "british_manifestos_domain":
        classes = BRITISH_MANIFESTOS_DOMAINS
    elif class_objective == "braulio":
        classes = BRAULIO_CLASSES
    elif class_objective == "braulio_subdomain":
        classes = SUBBRAULIO_CLASSES
    elif class_objective == "braulio_no_rebaba":
        classes = BRAULIO_CLASSES_NO_REBABA
    elif class_objective == "manifestos_domain":
        classes = GLOBAL_MANIFESTOS_DOMAINS
    elif class_objective == "manifestos_subdomain":
        classes = GLOBAL_MANIFESTOS_SUBDOMAINS
    return classes

def joint_previous_phrase_and_phrase_in_array(data_raw):
    joint_array = []
    for prev_phrase, phrase in zip(data_raw[1], data_raw):
        joint_array.append([prev_phrase, phrase])
    return joint_array

def load_sequential_sets(config):
    all_phrases = json.load(open(config['dataset_folder'] + config['dataset'],'r'))
    data_X = []
    data_y = []
    if config['class'] == 'british_manifestos_domain':
        class_obj  = "domain"
    for split in all_phrases:
        split_X = []
        split_y =[]
        for phrase in split:
            split_X.append(phrase['cleaned_phrase']) 
            split_y.append(phrase[class_obj])
        data_X.append(split_X)
        data_y.append(split_y)
    return [data_X], [data_y]

def load_embedding(config):
    model_path = EMB_FOLDER + config['embedding_name_1']
    if config['embedding_type_1'] == "word2vec":
        print "Is word2vec"
        print model_path
        try:
            model = Word2Vec.load(model_path)
        except Exception as e:
            print e
            model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif config['embedding_type_1'] == "txt":
        model = {}
        with open(model_path) as f:
            data = f.readlines()
            for row in data:
                split_word = row.split(",")
                word = split_word[0]
                embedding_values = split_word[1:-1]
                model[word.decode('utf-8')] = [float(i) for i in embedding_values]
                # print word
                # print embedding_values
                """print str(len(embedding_values ))
                print str(len(row))
                print row
                print multi_embedding[word]
                print str(type(multi_embedding[word][0]))"""
            del data, f
        gc.collect()
    elif config['embedding_type_1'] == "other":
        model = {}
        with open(model_path) as f:
            data = f.readlines()
            for row in data:
                word_plus_emb = row.split(" ")
                #print "Len of each row " + str(len(word_plus_emb))
                word = word_plus_emb[0]
                emb = word_plus_emb[1:-1]
                #print emb
                #print "Len of each emb is " + str(len(emb))
                #print str(emb)
                #print word
                model[word] = [float(i) for i in emb]
    else:
        print "Is FastText"
        #model = FastText.load_fasttext_format(model_path)
        if model_path.endswith('.bin'):
            model = FastText.load_fasttext_format(model_path)
        else:
            model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model
def generate_sequential_embedding_matrix(config, folder):
    model_path = EMB_FOLDER + config['embedding_name_1']
    print model_path
    data_X, data_y = load_sequential_sets(config)    
    print "We have a total of " + str(len(data_X[0])) + " phrases."
    #tweets = json.load(open('./datasets/tweets_elections_2015/2_tokenized/all_tweets.json','r'))
    #tweets_X = []
    #for tweet in tweets:
    #    tweets_X.append(tweet['cleaned_text'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([x for sublist in data_X[0] for x in sublist])
    #sequences_phrases = tokenizer.texts_to_sequences(data_X[0])
    #data_X[0] = pad_sequences(sequences_phrases, maxlen=max_phrase_length, padding='post')
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    model = load_embedding(config)
    weights = np.zeros((len(word_index) + 1, config['embedding_size_1']))
    unknown_words = {}
    for word, i in word_index.items():
        try:
            #pass
            if config['embedding_type_1'] != "other":
                embedding_vector = model[word]
            else:
                embedding_vector = model[word.decode('utf-8')]
            weights[i] = embedding_vector
        except Exception as e:
            #print e
            #print type(e) exceptions.KeyError
            if word in unknown_words:
                unknown_words[word] += 1
            else:
                unknown_words[word] = 1
    print "Number of unknown tokens: " + str(len(unknown_words))
    json.dump(unknown_words, open(folder + "/statistics/unknown_words" + model_path.split('/')[-1] + ".json", "w"))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights],  input_length=(config['max_phrase_length']), trainable=config['embedding_trainable_1'], mask_zero=True)
    return layer

def generate_sequential_sets(config, folder):
    classes_1 = get_classes_from_target_class(config['class'])
    data_X, data_y = load_sequential_sets(config)
    tokenizer = Tokenizer()
    o_train_splits = []
    o_eval_test_splits = []
    o_eval_splits = []
    o_test_splits = []
    tokenizer.fit_on_texts([x for sublist in data_X[0] for x in sublist])

    data_X_ids = []
    for split in data_X[0]:
        split_X = []
        #print split
        sequences_phrases = tokenizer.texts_to_sequences(split)
        #print sequences_phrases
        if not config['no_padding_for_lstms']:
            data_X_ids.append(np.array(pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'], padding='post')))
        else:
            data_X_ids.append(np.array(pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'])))
        #print data_X_ids[0]
    data_X_ids = [data_X_ids]
    sss_1 = ShuffleSplit(n_splits=1, test_size=0.3, random_state=config['seed'])
    train_X = generate_placeholder(len(data_X))
    train_y = generate_placeholder(len(data_y))
    eval_test_X = generate_placeholder(len(data_X))
    eval_test_y = generate_placeholder(len(data_y))
    print "Number of splits in the corpus: " + str(len(data_X_ids[0]))
    hashable_corpus = map(tuple, data_X[0])
    #print "Number of unique splits in the corpus: " + str(len(set(hashable_corpus)))
    print "Number of labels in the corpus: " + str(len(data_y))
    for train_indexes, eval_test_indexes in sss_1.split(data_X[0]):
        np.savetxt(folder + '/statistics/train_indexes' + config['class'] + '.out', train_indexes, delimiter=',')
        np.savetxt(folder + '/statistics/eval_test_indexes'+ config['class'] +'.out', eval_test_indexes, delimiter=',')
        for train_index in train_indexes:
            o_train_splits.append(data_X[0][train_index])
            for i in range(len(data_X_ids)):
                train_X[i].append(data_X_ids[i][train_index])
            for i in range(len(data_y)):
                train_y[i].append(data_y[i][train_index])
        for eval_test_index in eval_test_indexes:
            o_eval_test_splits.append(data_X[0][eval_test_index])
            for i in range(len(data_X_ids)):
                eval_test_X[i].append(data_X_ids[i][eval_test_index])
            for i in range(len(data_y)):
                eval_test_y[i].append(data_y[i][eval_test_index])
        print "Number of train indexes: " + str(len(train_indexes))
        print "Number of eval_test indexes: " + str(len(eval_test_indexes))
        print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((train_indexes, eval_test_indexes), axis=0))))
        print "Size of train: " + str(len(train_X[0]))
        print "Size of eval_test: " + str(len(eval_test_X[0]))
    """split_hashable = map(tuple, np.concatenate((train_X[0], eval_test_X[0]), axis=0))
    assert len(set(hashable_corpus)) == len(set(split_hashable))
    print "Number of unique phrases after the stratifying: " + str(len(set(split_hashable)))
    train_hashable = map(tuple, train_X[0])
    print "Number of unique phrases in train: " + str(len(set(train_hashable)))
    print "Number of unique phrases in eval_test: " + str(len(set(map(tuple, eval_test_X[0]))))"""
    """Split the 30% of the previous splitting into 50-50 (15-15) evaluation-test"""
    sss_2 = ShuffleSplit(n_splits=1, test_size=0.5, random_state=config['seed'])
    eval_X = generate_placeholder(len(data_X))
    eval_y = generate_placeholder(len(data_y))
    test_X = generate_placeholder(len(data_X))
    test_y = generate_placeholder(len(data_y))
    for eval_indexes, test_indexes in sss_2.split(eval_test_X[0]):
        np.savetxt(folder + '/statistics/eval_indexes' + config['class'] + '.out', eval_indexes, delimiter=',')
        np.savetxt(folder + '/statistics/test_indexes'+ config['class'] +'.out', test_indexes, delimiter=',')
        for eval_index in eval_indexes:
            for i in range(len(data_X)):
                eval_X[i].append(eval_test_X[i][eval_index])
            for i in range(len(data_y)):
                eval_y[i].append(eval_test_y[i][eval_index])
        for test_index in test_indexes:
            o_test_splits.append(o_eval_test_splits[test_index])
            for i in range(len(data_X)):
                test_X[i].append(eval_test_X[i][test_index])
            for i in range(len(data_y)):
                test_y[i].append(eval_test_y[i][test_index])
        """number_matches = 0
        for phrase in eval_X[0]:
            for phrase_2 in test_X[0]:
                if np.array_equal(phrase,phrase_2):
                    number_matches += 1
        print "Number of same phrases in eval test" + str(number_matches)
        print "Size of eval: " + str(len(eval_X[0]))
        print "Number of unique phrases in eval: " + str(len(set(map(tuple, eval_X[0]))))
        print "Size of test: " + str(len(test_X[0]))
        print "Number of unique phrases in test: " + str(len(set(map(tuple, test_X[0]))))"""
        print "Number of eval indexes: " + str(len(eval_indexes))
        print "Number of test indexes: " + str(len(test_indexes))
        print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((eval_indexes, test_indexes), axis=0))))
    if config['sequential_data'] and not config['sequential']:
        train_X[0] = flatten_array(train_X[0])
        train_y[0] = flatten_array(train_y[0])
        eval_X[0] = flatten_array(eval_X[0])
        eval_y[0] = flatten_array(eval_y[0])
        test_X[0] = flatten_array(test_X[0])
        test_y[0] = flatten_array(test_y[0])
        train_y[0] = to_one_hot_encoding(train_y[0], get_classes_from_target_class(config['class']))
        eval_y[0] = to_one_hot_encoding(eval_y[0], get_classes_from_target_class(config['class']))
        test_y[0] = to_one_hot_encoding(test_y[0], get_classes_from_target_class(config['class']))
    else:
        train_y[0] = to_one_hot_encoding(train_y[0], get_classes_from_target_class(config['class']), True)
        eval_y[0] = to_one_hot_encoding(eval_y[0], get_classes_from_target_class(config['class']), True)
        test_y[0] = to_one_hot_encoding(test_y[0], get_classes_from_target_class(config['class']), True)       
    print str(len(train_X[0]))
    print str(len(train_y[0]))
    print str(len(eval_X[0]))
    print str(len(eval_y[0]))
    print str(len(test_X[0]))
    print str(len(test_y[0]))
    print 'Lo de los teeses aita mesedez langudu'
    print test_X[0][0]
    print test_X[0][1]
    print test_X[0][2]
    return o_train_splits, train_X, train_y, eval_X, eval_y, o_test_splits, test_X, test_y



def add_prefix_to_sentence(emb_type, phrase, language):
    lang = LANGUAGES_DICT[language]
    new_cleaned_phrase = []
    for word in phrase:
        if emb_type == "other":
            new_cleaned_phrase.append(lang + "__" + word)
        else:
            new_cleaned_phrase.append(lang + ":" + word)
    return new_cleaned_phrase

def one_matrix_enconder(total_classes):
    encoder = {}
    array_size = len(total_classes)
    for i, code in enumerate(total_classes):
        vector =  np.random.rand(10,10)
        encoder[code] = [vector.tolist()]
    return encoder   

def load_json_data(data_path,previous_phrase, previous_previous, post_phrase, party, party_as_deconv, class_to_classify,
                   class_to_classify_2, party_as_rile_score, party_as_std_mean, mlingual, language=None, multiembedding=False, multiembeddingmultifile=False, emb_type=None):
    """
    It loads the data from a single file.
    """
    if party_as_deconv:
        party_encoder = one_matrix_enconder(POLITICAL_PARTIES)
    else:
        if not mlingual and language==None:
            party_encoder = one_hot_encoder(POLITICAL_PARTIES)
        else:
            party_encoder = one_hot_encoder(COUNTRY_PARTIES[language])
            
    print "Data path", data_path
    data = json.load(open(data_path, 'r'))
    data_X = []
    data_previous_phrase = []
    data_previous_previous = []
    data_post_phrase = []
    data_party = []
    data_phrase = []
    data_y = []
    data_y_2 = []
    print "Is previous phrase true?" + str(previous_phrase)
    print "Is previous previous phrase true?" + str(previous_previous)
    print "Is post phrase true?" + str(post_phrase)
    print "Is party true?" + str(party)
    for sentence in data:
        if 'domain' not in sentence and class_to_classify == 'manifestos_subdomain':
            subdomain = sentence['codes'][0].strip()
            if subdomain not in GLOBAL_MANIFESTOS_SUBDOMAINS:
                subdomain =   SUBSUB_TO_SUB[subdomain.replace("_",'.')] 
            if subdomain in BANNED_SUBS_FOR_HT:
                continue
        if party and not party_as_rile_score and not party_as_std_mean:
            if mlingual or language == 'english' and "v8" not in data_path:
                data_party.append(party_encoder[sentence['party_code']])
            else:
                data_party.append(party_encoder[sentence['party']])
        elif party and party_as_rile_score and not party_as_std_mean:
            data_party.append(sentence['rile_score'])
        elif party and not party_as_rile_score and party_as_std_mean:
            data_party.append(sentence['rile_score_mean_std'])
        if previous_phrase:
            if multiembedding or multiembeddingmultifile:
                data_previous_phrase.append(add_prefix_to_sentence(emb_type, sentence['previous_phrase'], language))
            else:
                if 'previous_phrase' in sentence:
                    data_previous_phrase.append(sentence['previous_phrase'])
                else:
                    if 'previous_tweet' in sentence:
                        data_previous_phrase.append(sentence['previous_tweet']['cleaned_text'])
                    else:
                        data_previous_phrase.append([])
        if previous_previous:
            if multiembedding or multiembeddingmultifile:
                data_previous_previous.append(add_prefix_to_sentence(emb_type, sentence['previous_phrase_2'], language))
            else:
                data_previous_previous.append(sentence['previous_phrase_2'])
        if post_phrase:
            if multiembedding or multiembeddingmultifile:
                data_post_phrase.append(add_prefix_to_sentence(emb_type, sentence['post_phrase'], language))
            else:
                data_post_phrase.append(sentence['post_phrase'])
        if multiembedding or multiembeddingmultifile:
            data_phrase.append(add_prefix_to_sentence(emb_type, sentence['cleaned_phrase'], language))
        else:
            if 'cleaned_phrase' in sentence:
                data_phrase.append(sentence['cleaned_phrase'])
            else:
                data_phrase.append(sentence['cleaned_text'])
            
        if class_to_classify == "complete":
            data_y.append(sentence["territorial"] + "_" + sentence['subdomain'])
        elif class_to_classify == "party":
            data_y.append(sentence['party'])
        else:
            if 'domain' not in sentence:
                if class_to_classify == 'manifestos_subdomain':
                    subdomain = sentence['codes'][0].strip()
                    if subdomain not in GLOBAL_MANIFESTOS_SUBDOMAINS:
                        subdomain =   SUBSUB_TO_SUB[subdomain.replace("_",'.')]
                    data_y.append(subdomain)
                else:
                    data_y.append(sentence['codes'][0].strip()[0])
            else:
                data_y.append(sentence[class_to_classify])
            if class_to_classify_2 != None:
                data_y_2.append(sentence[class_to_classify_2])
    print class_to_classify
    print type(data_y)
    print len(data_y)
    data_y = [data_y]
    print type(data_y)
    print data_y[0][0]
    if class_to_classify_2 != None:
        data_y.append(data_y_2)
    if party and not previous_phrase and not previous_previous and not post_phrase:
        return [data_phrase, data_party], data_y
    elif previous_phrase and not party and not previous_previous and not post_phrase:
        return [data_phrase, data_previous_phrase], data_y
    elif previous_phrase and previous_previous and not party and not post_phrase:
        return [data_phrase, data_previous_phrase, data_previous_previous], data_y
    elif party and previous_phrase and not previous_previous and not post_phrase:
        return [data_phrase, data_previous_phrase, data_party], data_y
    elif party and previous_phrase and previous_previous and not post_phrase:
        return [data_phrase, data_previous_phrase, data_previous_previous, data_party], data_y
    elif not party and not previous_phrase and not previous_previous and post_phrase:
        return [data_phrase, data_post_phrase], data_y
    elif not party and  previous_phrase and not previous_previous and post_phrase:
        return [data_phrase, data_previous_phrase, data_post_phrase], data_y
    elif party and  previous_phrase and not previous_previous and post_phrase:
        return [data_phrase, data_previous_phrase, data_post_phrase, data_party], data_y
    else:
        return [data_phrase], data_y
        #return [data_phrase,data_previous_phrase], data_y


def calculate_class_distribution(data_X, data_y, encoding, classes=None, binary=False):
    """
    Calculates and prints the class distribution of data
    """
    print "The length of data_X is " + str(len(data_X))
    print "The length of data_y is " + str(len(data_y))
    label_distribution = {}
    label_distribution_percent = {}
    for y in data_y:
        if encoding:
            if binary:
                y = str(find_1(y))
            else:
            #y = str(find_1(y) + 1)
                y = str(classes[find_1(y)])
        label_distribution.setdefault(y, 0)
        label_distribution[y] += 1
    for y_count in label_distribution:
        label_distribution_percent[y_count] = float(label_distribution[y_count])/float(len(data_X))
    print label_distribution
    print label_distribution_percent
    return label_distribution

def generate_placeholder(length):
    placeholder = []
    for i in range(length):
        placeholder.append([])
    return placeholder

def generate_embedding_matrix(config, folder, multilingua=None):
    model_path = EMB_FOLDER + config['embedding_name_1']
    print "Model path loading embedding", model_path
    if  config['topfish']:
        if config['multi_embedding']:
            all_splits_languages = []
            for config in multilingua:
                train, dev, test = load_topfish_datasets('./datasets/multilingual/' + config['language'] + "/2_tokenized/", config['language'], True)
                #all_splits_languages[config['language']] = {"train": train, "dev": dev, "test": test}
                all_splits_languages += train[0] + dev[0] + test[0]
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(all_splits_languages)
        else:
            print "master"
            train, dev, test = load_topfish_datasets('./datasets/multilingual/' + config['language'] + "/2_tokenized/", config['language'])
            all_splits = train[0] + dev[0] + test[0]
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(all_splits)
    else:
        if config['sequential_data']:
            data_X, data_y = load_sequential_sets(config)
            data_y = [flatten_array(data_y[0])]
            data_X = [flatten_array(data_X[0])]
        else:
            if config['multilingual'] and not config['multi_embedding']:
                data_path = config['dataset']
                data_X, data_y = load_dataset(data_path, config)
                on_fit = data_X[0]
            elif not config['multilingual'] and not config['multi_embedding'] and not config['multi_embedding_multi_file']:
                data_path = config['dataset_folder'] + config['dataset']
                data_X, data_y = load_dataset(data_path, config)
                on_fit = data_X[0]
            elif not config['multilingual'] and (config['multi_embedding'] or config['multi_embedding_multi_file']):
                data_path = config['dataset']
                data_X, data_y = load_dataset(data_path, config)
                on_fit = load_multilingua_datasets(multilingua)

        print "We have a total of " + str(len(data_X[0])) + " phrases."
        if config['add_annotated_tweets']:
            if config['class'] == 'braulio':
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio.json', 'r'))
                print "Adding tweets annotated with Braulios codification"
            elif config['class'] == 'braulio_no_rebaba':
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio_previous_tweet_no_rebaba.json', 'r'))
                print "Adding tweets annotated with Braulios codification and no rebaba"
            else:
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets.json','r'))
        elif config['add_english_annotated_tweets']:
            tweets = json.load(open('./datasets/english_annotated_tweets/english_tweets/2_tokenized/' + config['tweets_file'],'r'))
        elif config['add_non_annotated_tweets']:
            if config['year'] == '2015':
                print "Adding 2015 tweets"
                tweets = json.load(open('./datasets/tweets_elections/2015/2_tokenized/spanish_ge_2015_v2.json','r'))
            else:
                print "Adding 2016 tweets"
                tweets = json.load(open('./datasets/tweets_elections/2016/2_tokenized/spanish_ge_2016_v2.json','r'))

        if config['add_annotated_tweets'] or config['add_non_annotated_tweets'] or config['add_english_annotated_tweets']:
            tweets_X = []
            for tweet in tweets:
                tweets_X.append(tweet['cleaned_text'])
        tokenizer = Tokenizer()
        print type(data_y)
        print "0"
        print data_X[0][0]
        print "1"
        print data_X[0][1]
        print "2"
        print data_X[0][2]
        print "3"
        print data_X[0][3]
        if config['add_annotated_tweets'] or config['add_non_annotated_tweets'] or config['add_english_annotated_tweets']:
            print tweets_X[0]
            print "PASANDO"
            tokenizer.fit_on_texts(on_fit + tweets_X)
        else:
            tokenizer.fit_on_texts(on_fit)
        #sequences_phrases = tokenizer.texts_to_sequences(data_X[0])
        #data_X[0] = pad_sequences(sequences_phrases, maxlen=max_phrase_length, padding='post')
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    weights = np.zeros((len(word_index) + 1, config['embedding_size_1']))
    unknown_words = {}
    if config['multi_embedding_multi_file']:
        for lingua in multilingua:
            model = load_embedding(lingua)
            weights, unknown_words = populate_embedding_matrix(word_index, weights, unknown_words, model, config['language'])

    else:
        model = load_embedding(config)
        weights, unknown_words = populate_embedding_matrix(word_index, weights, unknown_words, model, None, config['embedding_type_1'])
    embeddings_to_tsv(word_index, weights)
    print "Number of unknown tokens: " + str(len(unknown_words))
    print "Model path when saving embedding unknown words", model_path
    print "Folder path when saving embedding unknown words", folder
    json.dump(unknown_words, open(folder + "/statistics/unknown_words" + model_path.split('/')[-1] + ".json", "w"))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights],  input_length=(config['max_phrase_length']), trainable=config['embedding_trainable_1'])
    return layer


def embeddings_to_tsv(word_index, weights):
    filename = "embeddings.tsv"
    words = np.full((len(word_index) + 1, 1), 'fill', dtype=object)
    words[:] = ' '
    for word, i in word_index.items():
        words[i] = word.encode('utf-8')
    with open(filename, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for weight in weights:
            tsv_writer.writerow(weight)
    with open("embeddings_name.tsv", 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for word in words:
            tsv_writer.writerow(word)

    #
    #for word,i in word_index.items():
    #    print str(i)
    #for word, embedding in word_count:
        #tsv_writer.writerow([word, count])


def populate_embedding_matrix(word_index, weights, unknown_words, model, language=None, emb_type=None):
    for word, i in word_index.items():
        if language != None:
            if word[0:3] == LANGUAGES_DICT[language] + ":":
                word = word.replace(LANGUAGES_DICT[language] + ":", "")
                weights, unknown_words = check_word_in_model(model, word, i, weights, unknown_words, emb_type)
        else:
            weights, unknown_words = check_word_in_model(model, word, i, weights, unknown_words, emb_type)
    return weights, unknown_words


def check_word_in_model(model, word, i, weights, unknown_words, emb_type):
    try:
        # pass
        if emb_type != "other":
            embedding_vector = model[word]
        else:
            embedding_vector = model[word.encode('utf-8')]
        weights[i] = embedding_vector
    except Exception as e:
        #print e
        #print type(e) exceptions.KeyError
        if word in unknown_words:
            unknown_words[word] += 1
        else:
            unknown_words[word] = 1
    return weights, unknown_words

def load_dataset(data_path, config):
    return load_json_data(data_path, config['previous_phrase'],
                   config['previous_previous'], config['post_phrase'], config['party'], config['party_as_deconv'], 
                   config['class'], config['class_2'], config['party_as_rile_score'], config['party_as_std_mean'],
                   config['multilingual'], config['language'], config['multi_embedding'],config['multi_embedding_multi_file'], config['embedding_type_1'])


def load_multilingua_datasets(configs):
    o_fit = []
    for config in configs:
        data_path = config['dataset']
        data_X, data_y = load_dataset(data_path, config)
        o_fit += data_X[0]
    return o_fit

def load_topfish_txt(path, language, add_prefix):
    print "paso"
    
    #print np.loadtxt(path)[0:10]
    loaded_json = json.load(open(path, 'r'))

    phrases = [[], []]
    for r in loaded_json:
        if add_prefix:
            phrases[0].append(add_prefix_to_sentence("other", r['cleaned_phrase'], language))
        else:
            phrases[0].append(r['cleaned_phrase'])
        phrases[1].append(r['domain'])
    return phrases

def load_topfish_datasets(path, language, add_prefix=None):
    splits = ['train', 'dev', 'test']
    splits_dict = {"train": [], "dev":[], "test": []}
    lang_abrev = LANGUAGES_DICT[language]
    for split in splits:
        full_path = path + lang_abrev + "-" +  split + ".txt"
        print full_path    
        splits_dict[split] = load_topfish_txt(full_path, language, add_prefix)
    return splits_dict['train'], splits_dict['dev'], splits_dict['test']



def generate_sets(config, folder,multilingua=None):
    classes_1 = get_classes_from_target_class(config['class'])
    classes_2 = get_classes_from_target_class(config['class_2'])
    if  config['topfish']:
        if config['multi_embedding']:
            all_splits_languages = []
            all_splits_languages_dict = {}
            all_train_x = []
            all_train_y = []
            all_dev_x = []
            all_dev_y = []
            for config in multilingua:
                train, dev, test = load_topfish_datasets('./datasets/multilingual/' + config['language'] + "/2_tokenized/", config['language'], True)
                test[1] = to_one_hot_encoding(test[1], get_classes_from_target_class(config['class']))
                all_splits_languages_dict[config['language']] = {"train": train, "dev": dev, "test": test}
                all_splits_languages += train[0] + dev[0] + test[0]
                all_train_x += train[0]
                all_train_y += train[1]
                all_dev_x += dev[0]
                all_dev_y += dev[1]
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(all_splits_languages)
            sequences_train = tokenizer.texts_to_sequences(all_train_x)
            sequences_dev = tokenizer.texts_to_sequences(all_dev_x)
            all_train_x = pad_sequences(sequences_train, maxlen=config['max_phrase_length'], padding='post')
            all_dev_x = pad_sequences(sequences_dev, maxlen=config['max_phrase_length'], padding='post')
            for config in multilingua:
                sequences_test = tokenizer.texts_to_sequences(all_splits_languages_dict[config['language']]['test'][0])
                all_splits_languages_dict[config['language']]['test'][0] = pad_sequences(sequences_test, maxlen=config['max_phrase_length'], padding='post')
            return [all_train_x], [to_one_hot_encoding(all_train_y, get_classes_from_target_class(config['class']))], [all_dev_x], [to_one_hot_encoding(all_dev_y, get_classes_from_target_class(config['class']))], all_splits_languages_dict
        else:
            print "master"
            train, dev, test = load_topfish_datasets('./datasets/multilingual/' + config['language'] + "/2_tokenized/", config['language'])
            all_splits = train[0] + dev[0] + test[0]
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(all_splits)
            sequences_train = tokenizer.texts_to_sequences(train[0])
            sequences_dev = tokenizer.texts_to_sequences(dev[0])
            sequences_test = tokenizer.texts_to_sequences(test[0])
            train[0] = pad_sequences(sequences_train, maxlen=config['max_phrase_length'], padding='post')
            dev[0] = pad_sequences(sequences_dev, maxlen=config['max_phrase_length'], padding='post')
            test[0] = pad_sequences(sequences_test, maxlen=config['max_phrase_length'], padding='post')
            return [train[0]],[to_one_hot_encoding(train[1], get_classes_from_target_class(config['class']))], [dev[0]], [to_one_hot_encoding(dev[1], get_classes_from_target_class(config['class']))], [test[0]], [to_one_hot_encoding(test[1], get_classes_from_target_class(config['class']))], [], [], []
    else:
        if config['multilingual'] and not config['multi_embedding']:
            data_path = config['dataset']
            data_X, data_y = load_dataset(data_path, config)
            on_fit = data_X[0]
        elif not config['multilingual'] and not config['multi_embedding'] and not config['multi_embedding_multi_file'] :
            data_path = config['dataset_folder'] + config['dataset']
            data_X, data_y = load_dataset(data_path, config)
            on_fit = data_X[0]
        elif not config['multilingual'] and (config['multi_embedding'] or config['multi_embedding_multi_file'] ):
            data_path = config['dataset']
            data_X, data_y = load_dataset(data_path, config)
            on_fit = load_multilingua_datasets(multilingua)


        print "We have a total of " + str(len(data_X[0])) + " phrases."
        if config['add_annotated_tweets']:
            if config['class'] == 'braulio':
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio.json', 'r'))
                print "Adding tweets annotated with Braulios codification"
            elif config['class'] == 'braulio_no_rebaba':
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio_previous_tweet_no_rebaba.json', 'r'))
                print "Adding tweets annotated with Braulios codification and no rebaba"
            else:
                tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets.json','r'))

        elif config['add_english_annotated_tweets']:
            tweets = json.load(open('./datasets/english_annotated_tweets/english_tweets/2_tokenized/' + config['tweets_file'],'r'))
        elif config['add_non_annotated_tweets']:
            if config['year'] == '2015':
                tweets = json.load(open('./datasets/tweets_elections/2015/2_tokenized/spanish_ge_2015_v2.json','r'))
            else:
                tweets = json.load(open('./datasets/tweets_elections/2016/2_tokenized/spanish_ge_2016_v2.json','r'))
        if config['add_annotated_tweets'] or config['add_english_annotated_tweets'] or config['add_non_annotated_tweets']:
            tweets_X = []
            for tweet in tweets:
                tweets_X.append(tweet['cleaned_text'])
        tokenizer = Tokenizer()
        "In case we wanna concatenate the previous phrase with the phrase instead of treating them as two different phrases"
        if config['concat_previous_phrase'] == True:
            new_data_X = []
            assert len(data_X[0]) == len(data_X[1])
            for sentence, pre in zip(data_X[0], data_X[1]):
                sentence =  pre + sentence
                new_data_X.append(sentence)
            data_X[0] = new_data_X
            #print data_X[0][2]
            #print data_X[1][2]
        print "0"
        print data_X[0][0]
        print "1"
        print data_X[0][1]
        print "2"
        print data_X[0][2]
        print "3"
        print data_X[0][3]
        o_phrases = data_X[0]#Original phrases without its convertion to indexes
        o_eval_test_phrases = []
        o_train_phrases = []
        o_test_phrases = []
        o_train_eval_phrases = []
        if config['previous_phrase']:
            o_prev_phrases = data_X[1]
            o_prev_eval_test_phrases = []
            o_prev_train_phrases = []
            o_prev_test_phrases = []
            o_prev_train_eval_phrases = []
        if config['add_annotated_tweets'] or config['add_english_annotated_tweets'] or config['add_non_annotated_tweets']:
            tokenizer.fit_on_texts(on_fit + tweets_X)
        else:
            tokenizer.fit_on_texts(on_fit)
        sequences_phrases = tokenizer.texts_to_sequences(data_X[0])
        if not config['no_padding_for_lstms']:
            data_X[0] = pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'], padding='post')
        else:
            data_X[0] = pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'])
        print "Just to see that concat is working fine"
        print o_phrases[0]
        print o_phrases[1]
        print str(len(data_X[0][1]))
        print data_X[0][1]
        if config['previous_phrase']:
            sequences_previous = tokenizer.texts_to_sequences(data_X[1])
            if not config['no_padding_for_lstms']:
                data_X[1] = pad_sequences(sequences_previous, maxlen=config['max_phrase_length'], padding='post')
            else:
                data_X[1] = pad_sequences(sequences_previous, maxlen=config['max_phrase_length'])
            if config['post_phrase']:
                sequences_posts = tokenizer.texts_to_sequences(data_X[2])
                if not config['no_padding_for_lstms']:
                    data_X[2] = pad_sequences(sequences_posts, maxlen=config['max_phrase_length'], padding='post')
                else:
                    data_X[2] = pad_sequences(sequences_posts, maxlen=config['max_phrase_length'])
        if config['previous_previous']:
            sequences_previous_previous = tokenizer.texts_to_sequences(data_X[2])
            if not config['no_padding_for_lstms']:
                data_X[2] = pad_sequences(sequences_previous_previous, maxlen=config['max_phrase_length'], padding='post')
            else:
                data_X[2] = pad_sequences(sequences_previous_previous, maxlen=config['max_phrase_length'])
        if config['post_phrase'] and not config['previous_phrase']:
            sequences_posts = tokenizer.texts_to_sequences(data_X[1])
            if not config['no_padding_for_lstms']:
                data_X[1] = pad_sequences(sequences_posts, maxlen=config['max_phrase_length'], padding='post')
            else:
                data_X[1] = pad_sequences(sequences_posts, maxlen=config['max_phrase_length'])
        if not config['cross_val_ready']:
            """Split dataset into stratified 70-30"""
            sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config['seed'])
            train_X = generate_placeholder(len(data_X))
            train_y = generate_placeholder(len(data_y))
            eval_test_X = generate_placeholder(len(data_X))
            eval_test_y = generate_placeholder(len(data_y))
            print "Number of phrases in the corpus: " + str(len(data_X[0]))
            hashable_corpus = map(tuple, data_X[0])
            print "Number of unique phrases in the corpus: " + str(len(set(hashable_corpus)))
            print "Number of labels in the corpus: " + str(len(data_y[0]))
            print "Calculating class distribution of the corpus"
            class_distribution = calculate_class_distribution(data_X[0], data_y[0], False)
            json.dump(class_distribution, open(folder + "/statistics/distribution.json", "w"))
            for train_indexes, eval_test_indexes in sss_1.split(data_X[0], data_y[0]):
                np.savetxt(folder + '/statistics/train_indexes' + config['class'] + '.out', train_indexes, delimiter=',')
                np.savetxt(folder + '/statistics/eval_test_indexes'+ config['class'] +'.out', eval_test_indexes, delimiter=',')
                for train_index in train_indexes:
                    o_train_phrases.append(o_phrases[train_index])
                    if config['previous_phrase']:
                        o_prev_train_phrases.append(o_prev_phrases[train_index])
                    for i in range(len(data_X)):
                        train_X[i].append(data_X[i][train_index])
                    for i in range(len(data_y)):
                        train_y[i].append(data_y[i][train_index])
                for eval_test_index in eval_test_indexes:
                    o_eval_test_phrases.append(o_phrases[eval_test_index])
                    if config['previous_phrase']:
                        o_prev_eval_test_phrases.append(o_prev_phrases[eval_test_index])
                    for i in range(len(data_X)):
                        eval_test_X[i].append(data_X[i][eval_test_index])
                    for i in range(len(data_y)):
                        eval_test_y[i].append(data_y[i][eval_test_index])
                print "Number of train indexes: " + str(len(train_indexes))
                print "Number of eval_test indexes: " + str(len(eval_test_indexes))
                print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((train_indexes, eval_test_indexes), axis=0))))
            split_hashable = map(tuple, np.concatenate((train_X[0], eval_test_X[0]), axis=0))
            assert len(set(hashable_corpus)) == len(set(split_hashable))
            print "Number of unique phrases after the stratifying: " + str(len(set(split_hashable)))
            train_hashable = map(tuple, train_X[0])
            print "Size of train: " + str(len(train_X[0]))
            print "Number of unique phrases in train: " + str(len(set(train_hashable)))
            print "Size of eval_test: " + str(len(eval_test_X[0]))
            print "Number of unique phrases in eval_test: " + str(len(set(map(tuple, eval_test_X[0]))))
            """Split the 30% of the previous splitting into 50-50 (15-15) evaluation-test"""
            sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['seed'])
            eval_X = generate_placeholder(len(data_X))
            eval_y = generate_placeholder(len(data_y))
            test_X = generate_placeholder(len(data_X))
            test_y = generate_placeholder(len(data_y))
            for eval_indexes, test_indexes in sss_2.split(eval_test_X[0], eval_test_y[0]):
                np.savetxt(folder + '/statistics/eval_indexes' + config['class'] + '.out', eval_indexes, delimiter=',')
                np.savetxt(folder + '/statistics/test_indexes'+ config['class'] +'.out', test_indexes, delimiter=',')
                for eval_index in eval_indexes:
                    for i in range(len(data_X)):
                        eval_X[i].append(eval_test_X[i][eval_index])
                    for i in range(len(data_y)):
                        eval_y[i].append(eval_test_y[i][eval_index])
                for test_index in test_indexes:
                    o_test_phrases.append(o_eval_test_phrases[test_index])
                    if config['previous_phrase']:
                        o_prev_test_phrases.append(o_prev_eval_test_phrases[test_index])
                    for i in range(len(data_X)):
                        test_X[i].append(eval_test_X[i][test_index])
                    for i in range(len(data_y)):
                        test_y[i].append(eval_test_y[i][test_index])
            """number_matches = 0
            for phrase in eval_X[0]:
                for phrase_2 in test_X[0]:
                    if np.array_equal(phrase,phrase_2):
                        number_matches += 1
            print "Number of same phrases in eval test" + str(number_matches)"""
            print "Size of eval: " + str(len(eval_X[0]))
            print "Number of unique phrases in eval: " + str(len(set(map(tuple, eval_X[0]))))
            print "Size of test: " + str(len(test_X[0]))
            print "Number of unique phrases in test: " + str(len(set(map(tuple, test_X[0]))))
            #print "Calculating class distribution of train with enconding"
            msg_1 = "Calculating class disitrbution of train without enconding"
            msg_2 = "Calculating class disitrbution of train with enconding"
            msg_3 = "Calculating class distribution of eval without encoding"
            msg_4 = "Calculating class distribution of eval with encoding"
            msg_5 = "Calculating class distribution of test without encoding"
            msg_6 = "Calculating class distribution of test with encoding"
            msg_7 = "Checking if one_hot_enconding_distribution is the same"
            if config['binary']:
                print msg_1
                distribution_without = calculate_class_distribution(train_X[0], train_y[0], False)
                if config['balance']:
                    train_X, train_y = balance_binary_distribution(train_X, train_y,class_to_train)
                    train_y[0] = to_binary(train_y[0], 1)
                else:
                    train_y[0] = to_binary(train_y[0], class_to_train)
                print msg_2 + " binary"
                distribution_with = calculate_class_distribution(train_X[0], train_y[0], True, binary=True)
                print msg_7
                check_one_hot_encoding_distribution(distribution_with, distribution_without, True, class_to_train)
                print msg_3
                calculate_class_distribution(eval_X[0], eval_y[0], False)
                print msg_4 + " binary"
                if config['balance']:
                    eval_X, eval_y = balance_binary_distribution(eval_X, eval_y,class_to_train)
                    eval_y [0]= to_binary(eval_y[0], 1)
                else:
                    eval_y [0]= to_binary(eval_y[0], class_to_train)
                calculate_class_distribution(eval_X[0], eval_y[0], True, binary=True)
                print msg_5
                calculate_class_distribution(test_X[0], test_y[0], False)
                print msg_6 + " binary"
                test_y[0] = to_binary(test_y[0], class_to_train)
                calculate_class_distribution(test_X[0], test_y[0], True, binary=True)
            else:
                print msg_1
                distribution_without = calculate_class_distribution(train_X[0], train_y[0], False)
                train_y[0] = to_one_hot_encoding(train_y[0], classes_1)
                print msg_2
                distribution_with = calculate_class_distribution(train_X[0], train_y[0], True, classes_1)
                print msg_7
                check_one_hot_encoding_distribution(distribution_with, distribution_without)
                print msg_3
                calculate_class_distribution(eval_X[0], eval_y[0], False)
                print msg_4
                eval_y[0] = to_one_hot_encoding(eval_y[0],classes_1)
                calculate_class_distribution(eval_X[0], eval_y[0], True,classes_1)
                print msg_5
                calculate_class_distribution(test_X[0], test_y[0], False)
                print msg_6
                test_y[0] = to_one_hot_encoding(test_y[0], classes_1)
                calculate_class_distribution(test_X[0], test_y[0], True, classes_1)
                if classes_2 != None:
                    print "Classes 2 are inserted"
                    train_y[1] = to_one_hot_encoding(train_y[1], classes_2)
                    eval_y[1] = to_one_hot_encoding(eval_y[1], classes_2)
                    test_y[1] = to_one_hot_encoding(test_y[1], classes_2)
            #train_y = to_one_hot_encoding(train_y, classes)
            check_length_x_y_equal(train_X[0], train_y[0])
            check_length_x_y_equal(eval_X[0], eval_y[0])
            check_length_x_y_equal(test_X[0], test_y[0])
            print 'Lo de los teeses aita mesedez langudu'
            print o_test_phrases[0]
            print o_test_phrases[1]
            print o_test_phrases[2]
            if config['previous_phrase']:
                return train_X, train_y, eval_X, eval_y, test_X, test_y,o_train_phrases, o_test_phrases, o_prev_test_phrases
            else:
                return train_X, train_y, eval_X, eval_y, test_X, test_y, o_train_phrases, o_test_phrases, []
        else:
            sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config['seed'])
            train_eval_X = generate_placeholder(len(data_X))
            train_eval_y = generate_placeholder(len(data_y))
            test_X = generate_placeholder(len(data_X))
            test_y = generate_placeholder(len(data_y))
            print "Number of phrases in the corpus: " + str(len(data_X[0]))
            hashable_corpus = map(tuple, data_X[0])
            print "Number of unique phrases in the corpus: " + str(len(set(hashable_corpus)))
            print "Number of labels in the corpus: " + str(len(data_y[0]))
            print "Calculating class distribution of the corpus"
            class_distribution = calculate_class_distribution(data_X[0], data_y[0], False)
            json.dump(class_distribution, open(folder + "/statistics/distribution.json", "w"))
            for train_eval_indexes, test_indexes in sss_1.split(data_X[0], data_y[0]):
                np.savetxt(folder + '/statistics/train_eval_indexes' + config['class'] + '.out', train_eval_indexes, delimiter=',')
                np.savetxt(folder + '/statistics/test_indexes'+ config['class'] +'.out', test_indexes, delimiter=',')
                for train_eval_index in train_eval_indexes:
                    o_train_eval_phrases.append(o_phrases[train_eval_index])
                    for i in range(len(data_X)):
                        train_eval_X[i].append(data_X[i][train_eval_index])
                    for i in range(len(data_y)):
                        train_eval_y[i].append(data_y[i][train_eval_index])
                for test_index in test_indexes:
                    o_test_phrases.append(o_phrases[test_index])
                    for i in range(len(data_X)):
                        test_X[i].append(data_X[i][test_index])
                    for i in range(len(data_y)):
                        test_y[i].append(data_y[i][test_index])
                print "Number of train indexes: " + str(len(train_eval_indexes))
                print "Number of eval_test indexes: " + str(len(test_indexes))
                print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((train_eval_indexes, test_indexes), axis=0))))
            split_hashable = map(tuple, np.concatenate((train_eval_X[0], test_X[0]), axis=0))
            assert len(set(hashable_corpus)) == len(set(split_hashable))
            print "Number of unique phrases after the stratifying: " + str(len(set(split_hashable)))
            train_hashable = map(tuple, train_eval_X[0])
            print "Size of train: " + str(len(train_eval_X[0]))
            print "Number of unique phrases in train: " + str(len(set(train_hashable)))
            print "Size of eval_test: " + str(len(test_X[0]))
            print "Number of unique phrases in test: " + str(len(set(map(tuple, test_X[0]))))
            test_y[0] = to_one_hot_encoding(test_y[0], classes_1)
            if classes_2 != None:
                test_y[1] = to_one_hot_encoding(test_y[1], classes_2)
            skf = StratifiedKFold(n_splits=5, random_state=config['seed'], shuffle=True)
            train_folds_X = []
            o_train_folds = []
            eval_folds_X = []
            o_eval_folds = []
            train_folds_y = []
            eval_folds_y = []
            print type(train_eval_X)
            for train_indexes, eval_indexes in skf.split(train_eval_X[0], train_eval_y[0]):
                print train_indexes[0]
                print "Length of train eval " + str(len(train_eval_X[0]))
                print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((train_indexes, eval_indexes), axis=0))))
                tmp_train_X = generate_placeholder(len(data_X))
                tmp_o_train = []
                tmp_train_y = generate_placeholder(len(data_y))
                tmp_eval_X = generate_placeholder(len(data_X))
                tmp_o_eval = []
                tmp_eval_y = generate_placeholder(len(data_y))
                for train_index in train_indexes:
                    tmp_o_train.append(o_train_eval_phrases[train_index])
                    for i in range(len(data_X)):
                        tmp_train_X[i].append(train_eval_X[i][train_index])
                    for i in range(len(data_y)):
                        tmp_train_y[i].append(train_eval_y[i][train_index])
                for eval_index in eval_indexes:
                    tmp_o_eval.append(o_train_eval_phrases[eval_index])
                    for i in range(len(data_X)):
                        tmp_eval_X[i].append(train_eval_X[i][eval_index])
                    for i in range(len(data_y)):
                        tmp_eval_y[i].append(train_eval_y[i][eval_index])
                tmp_eval_y[0] = to_one_hot_encoding(tmp_eval_y[0], classes_1)
                tmp_train_y[0] = to_one_hot_encoding(tmp_train_y[0], classes_1)
                if classes_2 != None:
                    print "Classes 2 are inserted"
                    tmp_train_y[1] = to_one_hot_encoding(tmp_train_y[1], classes_2)
                    tmp_eval_y[1] = to_one_hot_encoding(tmp_eval_y[1], classes_2)
                train_folds_X.append(tmp_train_X)
                o_train_folds.append(tmp_o_train)
                train_folds_y.append(tmp_train_y)
                eval_folds_X.append(tmp_eval_X)
                o_eval_folds.append(tmp_o_eval)
                eval_folds_y.append(tmp_eval_y)
                #print str(len(train_index))
                #print str(len(eval_index))
            """print len(train_folds_X[0][0])
            print len(o_train_folds[0])
            print len(train_folds_y[0][0])
            print len(eval_folds_X[0][0])
            print len(o_eval_folds[0])
            print len(eval_folds_y[0][0])"""
            #return "BREAK"
            """iguales = 0
            for p in o_eval_folds[0]:
                for p_2 in o_eval_folds[1]:
                    if p  == p_2:
                        iguales = iguales + 1
            print "Numero de iguales" + str(iguales)"""
            print "TEST SOME VALUES TO SEE THAT ALWAYS ARE THE SAME"
            print o_test_phrases[0]
            print o_test_phrases[11]
            print o_test_phrases[25]
            return train_folds_X, o_train_folds, train_folds_y, eval_folds_X,o_eval_folds, eval_folds_y, o_test_phrases, test_X, test_y


def compute_weight_classes(classes, sample_y):
    sample_y_labels = []
    for label in sample_y:
        sample_y_labels.append(classes[find_1(label)])

    class_weight_values = class_weight.compute_class_weight('balanced',classes , sample_y_labels)
    class_weight_values_dict = {}
    for value, code in zip(class_weight_values, classes):
        class_weight_values_dict[code] = value
    print class_weight_values
    return class_weight_values

def generate_word2vec_corpus():
    phrases = json.load(open('./datasets/rmp_dataset/2_tokenized/all_phrases_previous_party.json','r'))
    corpus = ''
    for phrase in phrases:
        corpus += (' '.join(phrase['cleaned_phrase']) + "\n").encode('utf-8')
    with open("./models/phrase_per_line.txt", "w") as text_file:
            text_file.write(corpus)
if __name__ == '__main__':
    generate_word2vec_corpus()

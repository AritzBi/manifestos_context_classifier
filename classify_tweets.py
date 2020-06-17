import json, copy
import numpy as np
from evaluation import evaluate_model
from data_preprocessing import to_one_hot_encoding, get_classes_from_target_class, load_json_data, one_hot_encoder
from keras.preprocessing.sequence import pad_sequences
from mytext import Tokenizer
from keras.models import load_model
from constants import SUBBRAULIO_CLASSES, GLOBAL_MANIFESTOS_SUBDOMAINS, DOMAIN_CLASSES, SUBSUB_TO_SUB, COUNTRY_PARTIES
from fast_scripts import generate_placeholder
from sklearn.model_selection import StratifiedShuffleSplit
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification

def get_english_annotated_tweets_sets(config, folder, only_tokenizer=False):
    samples_limit_for_stratification = 8
    print config
    path = "./datasets/english_annotated_tweets/english_tweets/2_tokenized/" + config['tweets_file']
    annotated_tweets = json.load(open(path, 'r'))
    print "Total number of annotated english tweets is", len(annotated_tweets)
    print "Sample tweet", annotated_tweets[0]
    samples_distribution_domain_all = {}
    samples_distribution_subdomain_all = {}
    samples_distribution_domain = {}
    samples_distribution_subdomain = {}
    total_codes = 0
    for tweet in annotated_tweets:
        tweet['domain_codes'] = []
        tweet['subdomain_codes'] = []
        for code in tweet['codes']:
            total_codes += 1
            code = code.strip()
            if code not in samples_distribution_subdomain_all:
                samples_distribution_subdomain_all[code] = 1
            else:
                samples_distribution_subdomain_all[code] += 1
            subdomain = code
            if subdomain not in GLOBAL_MANIFESTOS_SUBDOMAINS:
                #print tweet
                subdomain = SUBSUB_TO_SUB[subdomain.replace('_','.')]
            domain = code[0]
            tweet['domain_codes'].append(domain)
            tweet['subdomain_codes'].append(subdomain)
            if 'domain' not in tweet:
                tweet['domain'] = domain
                tweet['subdomain'] = subdomain
                if subdomain not in samples_distribution_subdomain:
                    samples_distribution_subdomain[subdomain] = 1
                else:
                    samples_distribution_subdomain[subdomain] += 1
                if domain not in samples_distribution_domain:
                    samples_distribution_domain[domain] = 1
                else:
                    samples_distribution_domain[domain] += 1
            if domain not in DOMAIN_CLASSES:
                print tweet
            if domain not in samples_distribution_domain_all:
                samples_distribution_domain_all[domain] = 1
            else:
                samples_distribution_domain_all[domain] += 1
    """print "Annotated tweets distribution taking into account multi-label annotation-----------------------------------"
    for c in DOMAIN_CLASSES:
        print("Domain %s; Number of samples:  %s; Percentage: %.2f " % (c, samples_distribution_domain_all[c], samples_distribution_domain_all[c]/float(total_codes)*100))
    for c in GLOBAL_MANIFESTOS_SUBDOMAINS:
        if c in samples_distribution_subdomain_all:
            print("Subdomain %s; Number of samples: %s; Percentage: %.2f " % (c, samples_distribution_subdomain_all[c],samples_distribution_subdomain_all[c]/float(total_codes)*100))
    print "-----------------------------------------------------------------------------------------------------------" """
    print "Annotated tweets distribution with multiclass-annotation"
    banned_codes_for_classification = []
    for c in DOMAIN_CLASSES:
        print("Domain %s; Number of samples:  %s; Percentage: %.2f " % (c, samples_distribution_domain[c], samples_distribution_domain[c]/float(len(annotated_tweets))*100))
    for c in GLOBAL_MANIFESTOS_SUBDOMAINS:
        if c in samples_distribution_subdomain:
            print("Subdomain %s; Number of samples: %s; Percentage: %.2f " % (c, samples_distribution_subdomain[c],samples_distribution_subdomain[c]/float(len(annotated_tweets))*100))
            if samples_distribution_subdomain[c] < samples_limit_for_stratification:
                print "NOT ENOUGH SAMPLES!!!!"
                banned_codes_for_classification.append(c)
    print "BANNNED!!!!!!!!!!!!!!!!"
    print banned_codes_for_classification
    if config['party'] and config['previous_phrase']:
        tweets_X = [[], [], []]
    elif config['party'] or config['previous_phrase']:
        tweets_X = [[], []]
    else:
        tweets_X = [[]]
    tweets_y = []
    party_encoder = one_hot_encoder(COUNTRY_PARTIES['english'])
    for tweet in annotated_tweets:
        if config['architecture'] == 'multi_label':
            class_to_pick = 'domain_codes'
        else:
            class_to_pick = config['class']
        if config['class'] == 'manifestos_subdomain':
            if config['architecture'] == 'multi_label':
                class_to_pick = 'subdomain_codes'
            else:
                 class_to_pick = 'subdomain'
        if config['architecture'] == 'multi_label':
            tweets_y.append(multilabel_array_to_onehot(config['class'], tweet[class_to_pick]))
        else:
            tweets_y.append(tweet[class_to_pick])
        tweets_X[0].append(tweet['cleaned_text'])
        if config['previous_phrase']:
            if 'previous_tweet' in tweet:
                tweets_X[1].append(tweet['previous_tweet']['cleaned_text'])
            else:
                tweets_X[1].append([])
            if config['party']:
                tweets_X[2].append(party_enconder[tweet['party']])
        elif config['party']:
            tweets_X[1].append(party_encoder[tweet['party']])    
    tweets_y = [tweets_y]
    o_phrases = tweets_X[0]#Original phrases without its convertion to indexes
    o_eval_test_phrases = []
    o_train_phrases = []
    o_test_phrases = []
    o_train_eval_phrases = []
    if config['previous_phrase']:
        o_prev_phrases = tweets_X[1]
        o_prev_eval_test_phrases = []
        o_prev_train_phrases = []
        o_prev_test_phrases = []
        o_prev_train_eval_phrases = []
            
    data_X, data_y = load_json_data( config['dataset_folder'] + config['dataset'], config['previous_phrase'], config['previous_previous'], config['post_phrase'],config['party'], config['party_as_deconv'], config['class'], config['class_2'], False, False, False, config['language'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_X[0] + tweets_X[0])
    if only_tokenizer:
        return tokenizer
    sequences_phrases = tokenizer.texts_to_sequences(tweets_X[0])
    if config['previous_phrase']:
        prev_phrases = tokenizer.texts_to_sequences(tweets_X[1])
        if config['party']:
            tweets_X_party = tweets_X[2]
    elif config['party']:
        tweets_X_party = tweets_X[1]
    if not config['no_padding_for_lstms']:
        tweets_X = pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'], padding='post')
        if config['previous_phrase']:
            tweets_X_prev = pad_sequences(prev_phrases, maxlen=config['max_phrase_length'], padding='post')    
    else:
        tweets_X = pad_sequences(sequences_phrases, maxlen=config['max_phrase_length'])
        if config['previous_phrase']:
            tweets_X_prev = pad_sequences(prev_phrases, maxlen=config['max_phrase_length']) 
    tweets_X_tmp = []
    tweets_X_prev_tmp = []
    tweets_X_party_tmp = []
    tweets_y_tmp = []
    o_phrases_tmp = []
    o_prev_phrases_tmp = []
    if config['previous_phrase']:
        if not config['party']:
            for tweet, prev_tweet, y_label, o_phrase, o_prev_phrase in zip(tweets_X, tweets_X_prev, tweets_y[0], o_phrases, o_prev_phrases):
                if y_label in banned_codes_for_classification:
                    continue
                tweets_X_tmp.append(tweet)
                tweets_X_prev_tmp.append(prev_tweet)
                tweets_y_tmp.append(y_label)
                o_phrases_tmp.append(o_phrase)
                o_prev_phrases_tmp.append(o_prev_phrase)
        else:
            for tweet, prev_tweet, party, y_label, o_phrase, o_prev_phrase in zip(tweets_X, tweets_X_prev,tweets_X_party, tweets_y[0], o_phrases, o_prev_phrases):
                if y_label in banned_codes_for_classification:
                    continue
                tweets_X_tmp.append(tweet)
                tweets_X_prev_tmp.append(prev_tweet)
                tweets_X_party_tmp.append(party)
                tweets_y_tmp.append(y_label)
                o_phrases_tmp.append(o_phrase)
                o_prev_phrases_tmp.append(o_prev_phrase)
    elif config['party'] :
        for tweet, party,  y_label, o_phrase in zip(tweets_X, tweets_X_party, tweets_y[0], o_phrases):
            if y_label in banned_codes_for_classification:
                continue
            tweets_X_tmp.append(tweet)
            tweets_X_party_tmp.append(party)
            tweets_y_tmp.append(y_label)
            o_phrases_tmp.append(o_phrase)  
    else:
        for tweet, y_label, o_phrase in zip(tweets_X, tweets_y[0], o_phrases):
            if y_label in banned_codes_for_classification:
                continue
            tweets_X_tmp.append(tweet)
            tweets_y_tmp.append(y_label)
            o_phrases_tmp.append(o_phrase)
    
    tweets_X = [tweets_X_tmp]
    if config['previous_phrase']:
        tweets_X.append(tweets_X_prev_tmp)
    if config['party']:
        tweets_X.append(tweets_X_party_tmp)
    
    tweets_y = [tweets_y_tmp]
    o_phrases = o_phrases_tmp
    if config['previous_phrase']:
        o_prev_phrases = o_prev_phrases_tmp
    tweets_train_X = generate_placeholder(len(tweets_X))
    tweets_train_y = generate_placeholder(len(tweets_y))
    tweets_eval_test_X = generate_placeholder(len(tweets_X))
    tweets_eval_test_y = generate_placeholder(len(tweets_y))
    if config['big_test']:
        sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['seed'])
        train_indexes, eval_test_indexes = next(sss_1.split(tweets_X[0], tweets_y[0]))
    elif not config['architecture'] == 'multi_label':
        sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config['seed'])
        train_indexes, eval_test_indexes = next(sss_1.split(tweets_X[0], tweets_y[0]))
    elif config['architecture'] == 'multi_label':
        sss_1 = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.3, 0.7], random_state=config['seed'] )
        train_indexes, eval_test_indexes = next(sss_1.split(np.array(tweets_X[0]), np.array(tweets_y[0])))
    #for train_indexes, eval_test_indexes in sss_1.split(tweets_X[0], tweets_y[0]):
    np.savetxt(folder + '/statistics/train_indexes' + config['class'] + '.out', train_indexes, delimiter=',')
    np.savetxt(folder + '/statistics/eval_test_indexes'+ config['class'] +'.out', eval_test_indexes, delimiter=',')
    for train_index in train_indexes:
        o_train_phrases.append(o_phrases[train_index])
        if config['previous_phrase']:
            o_prev_train_phrases.append(o_prev_phrases[train_index])
        for i in range(len(tweets_X)):
            tweets_train_X[i].append(tweets_X[i][train_index])
        for i in range(len(tweets_y)):
            tweets_train_y[i].append(tweets_y[i][train_index])
    for eval_test_index in eval_test_indexes:
        o_eval_test_phrases.append(o_phrases[eval_test_index])
        if config['previous_phrase']:
            o_prev_eval_test_phrases.append(o_prev_phrases[eval_test_index])
        for i in range(len(tweets_X)):
            tweets_eval_test_X[i].append(tweets_X[i][eval_test_index])
        for i in range(len(tweets_y)):
            tweets_eval_test_y[i].append(tweets_y[i][eval_test_index])
    print "Number of train indexes: " + str(len(train_indexes))
    print "Number of eval_test indexes: " + str(len(eval_test_indexes))
    print "Number of unique indexes after the stratifying", str(len(np.unique(np.concatenate((train_indexes, eval_test_indexes), axis=0))))
    tweets_eval_X = generate_placeholder(len(tweets_X))
    tweets_eval_y = generate_placeholder(len(tweets_y))
    tweets_test_X = generate_placeholder(len(tweets_X))
    tweets_test_y = generate_placeholder(len(tweets_y))
    """Split the 30% of the previous splitting into 50-50 (15-15) evaluation-test"""
    if config['architecture'] == 'multi_label':
        sss_2 = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5], random_state=config['seed'] )
        eval_indexes, test_indexes = next(sss_2.split(np.array(tweets_eval_test_X[0]), np.array(tweets_eval_test_y[0])))
    else:
        sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['seed'])
        eval_indexes, test_indexes = next(sss_2.split(tweets_eval_test_X[0], tweets_eval_test_y[0]))
    #for eval_indexes, test_indexes in sss_2.split(tweets_eval_test_X[0], tweets_eval_test_y[0]):
    np.savetxt(folder + '/statistics/eval_indexes' + config['class'] + '.out', eval_indexes, delimiter=',')
    np.savetxt(folder + '/statistics/test_indexes'+ config['class'] +'.out', test_indexes, delimiter=',')
    for eval_index in eval_indexes:
        for i in range(len(tweets_X)):
            tweets_eval_X[i].append(tweets_eval_test_X[i][eval_index])
        for i in range(len(tweets_y)):
            tweets_eval_y[i].append(tweets_eval_test_y[i][eval_index])
    for test_index in test_indexes:
        o_test_phrases.append(o_eval_test_phrases[test_index])
        if config['previous_phrase']:
            o_prev_test_phrases.append(o_prev_eval_test_phrases[test_index])
        for i in range(len(tweets_X)):
            tweets_test_X[i].append(tweets_eval_test_X[i][test_index])
        for i in range(len(tweets_y)):
            tweets_test_y[i].append(tweets_eval_test_y[i][test_index])
    #tweets_y = to_one_hot_encoding(tweets_y, get_classes_from_target_class(config['class']))
    if config['architecture'] != 'multi_label':
        tweets_train_y[0] = to_one_hot_encoding(tweets_train_y[0], get_classes_from_target_class(config['class']))
        tweets_eval_y[0] = to_one_hot_encoding(tweets_eval_y[0], get_classes_from_target_class(config['class']))
        tweets_test_y[0] = to_one_hot_encoding(tweets_test_y[0], get_classes_from_target_class(config['class']))
    print '------------------ PADRE LO DE LOS TUITS TRAIN -------------------------------'
    print tweets_train_X[0][0]
    print tweets_train_X[0][1]
    print tweets_train_X[0][2]
    print '------------------ PADRE LO DE LOS TUITS EVAL -------------------------------'
    print tweets_eval_X[0][0]
    print tweets_eval_X[0][1]
    print tweets_eval_X[0][2]
    print '------------------ PADRE LO DE LOS TUITS TESTS -------------------------------'
    print o_test_phrases[0]
    print o_test_phrases[1]
    print o_test_phrases[2]
    if config['big_test']:
        return  tweets_eval_X, tweets_eval_y, tweets_test_X, tweets_test_y, tweets_train_X, tweets_train_y
    else:
        return tweets_train_X, tweets_train_y, tweets_eval_X, tweets_eval_y, tweets_test_X, tweets_test_y

def handle_annotated_english_tweets(config, folder):
    tweets_train_X, tweets_train_y, tweets_eval_X, tweets_eval_y, tweets_test_X, tweets_test_y = get_english_annotated_tweets_sets(config, folder)
    softmax_values, predicted_classes = evaluate_model(config, folder, tweets_test_X, tweets_test_y, None, None, True)

def classify_non_annotated_tweets(config, folder):
    non_annotated_tweets = json.load(open('./datasets/english_annotated_tweets/english_non_annotated_tweets/2_tokenized/non_annotated_candidates_tweets.json', 'r'))
    tokenizer = get_english_annotated_tweets_sets(config, folder, True)
    party_encoder = one_hot_encoder(COUNTRY_PARTIES['english'])
    tweets_text = []
    tweets_party = []
    for tweet in non_annotated_tweets:
        tweets_text.append(tweet['cleaned_text'])
        tweets_party.append(party_encoder[tweet['party']])
    tweets_text = tokenizer.texts_to_sequences(tweets_text)
    if not config['no_padding_for_lstms']:
        tweets_text = pad_sequences(tweets_text, maxlen=config['max_phrase_length'], padding='post')
    else:
        tweets_text = pad_sequences(tweets_text, maxlen=config['max_phrase_length'])
    tweets_X = [tweets_text, tweets_party]
    model = load_model(folder + "best_model.hdf5")
    x_dict = {"phrases": np.array(tweets_X[0]), "party": np.array(tweets_X[1])}
    predicted_y = model.predict(x_dict)
    print str(len(predicted_y))
    predicted_y = predicted_y.argmax(axis=-1)
    raw_predicted_y = []
    for label in predicted_y:
        raw_predicted_y.append(get_classes_from_target_class(config['class'])[label])
    if config['class'] == 'domain':
        tweets_per_party = {'61320':{'count': {'total': 0, 'domain': { '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7':0 }}},'61620':{'count': {'total': 0, 'domain': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7':0 }}}}
    else:
        empty_manifestos_subdomain_dict = {}
        for subd in GLOBAL_MANIFESTOS_SUBDOMAINS:
            empty_manifestos_subdomain_dict[subd] = 0
        tweets_per_party = {'61320':{'count': {'total': 0, 'domain': empty_manifestos_subdomain_dict }},'61620':{'count': {'total': 0, 'domain':  copy.deepcopy(empty_manifestos_subdomain_dict) }}}
    for tweet, label in zip(non_annotated_tweets,raw_predicted_y):
        tweets_per_party[tweet['party']]['count']['domain'][label] += 1
        tweets_per_party[tweet['party']]['count']['total'] += 1
    print tweets_per_party
    for party in tweets_per_party:
        tweets_per_party[party]['percent'] = {}
        for dom in tweets_per_party[party]['count']['domain']:
            tweets_per_party[party]['percent'][dom] = (float(tweets_per_party[party]['count']['domain'][dom])/float(tweets_per_party[party]['count']['total']) * 100)
    print tweets_per_party

    for party in tweets_per_party:
        print party
        print '-------------------------------------------------------'
        for w in sorted(tweets_per_party[party]['percent'], key=tweets_per_party[party]['percent'].get, reverse=True):
            print w, tweets_per_party[party]['percent'][w]


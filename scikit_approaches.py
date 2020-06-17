from data_preprocessing import generate_sets, find_1
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from evaluation import generate_results
import json
def create_classifer(config=None, class_weight_values=None):
    if config['classifier']=='linear_svm':
        return LinearSVC(C=1, class_weight=class_weight_values, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0)
    else:
        return SVC(class_weight=class_weight_values)

def classic(config, folder, classes):
    concat_previous_phrase = False
    if config['previous_phrase']:
        config['concat_previous_phrase'] = True
    else:
        config['concat_previous_phrase'] = False
    if config['cross_val_ready']:
        generate_sets(config, folder)
    else:
        train_X, train_y, eval_X, eval_y, test_X, test_y,o_train_phrases, o_test_phrases, o_prev_test_phrases = generate_sets(config,
                                                                                                          folder)
        train_X = []
        for p in o_train_phrases:
            #print p
            train_X.append(' '.join(p))
        test_X = []
        for p in o_test_phrases:
            print p
            test_X.append(' ' .join(p))
        train_y_labels = []
        for label in train_y[0]:
            train_y_labels.append(classes[find_1(label)])
        print train_y_labels[0:10]
        test_y_labels = []
        for label in test_y[0]:
            test_y_labels.append(classes[find_1(label)])
        class_weight_values_dict = {}
        print "Train_x" + str(len(train_X))
        print "Train_y_labels" + str(len(train_y_labels))
        print "Train_x" + str(len(test_X))
        print "Train_y_labels" + str(len(test_y_labels))
        if config['weight_class']:
                class_weight_values = class_weight.compute_class_weight('balanced',classes , test_y_labels)
                class_weight_values_dict = {}
                for value, code in zip(class_weight_values, classes):
                    class_weight_values_dict[code] = value
                print "Balancing classes"
                print class_weight_values
        vectorizer_1 = CountVectorizer(ngram_range=(1,3))
        matrix = vectorizer_1.fit_transform(train_X)
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(matrix)
        clf = create_classifer(config, class_weight_values_dict)
        clf.fit(tfidf, train_y_labels)
        print "Evaluating with test set"
        test_matrix = vectorizer_1.transform(test_X)
        tfidf_test = tfidf_transformer.transform(test_matrix)
        acc = accuracy_score(test_y_labels, clf.predict(tfidf_test))
        f1_macro = f1_score(test_y_labels, clf.predict(tfidf_test), average="macro")
        results_processed = generate_results(test_y_labels, clf.predict(tfidf_test), classes, folder)
        #print results_processed
        print str(acc)
        print str(f1_macro)
        #acc, f1_macro, results_processed

        tweets = json.load(open('./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio.json','r'))
        tweets_X = []
        tweets_y = []
        for tweet in tweets:
            tweets_X.append(' '.join(tweet['cleaned_text']))
            tweets_y.append(tweet[config['class']])
        test_matrix = vectorizer_1.transform(tweets_X)
        tfidf_test = tfidf_transformer.transform(test_matrix)
        acc = accuracy_score(tweets_y, clf.predict(tfidf_test))
        f1_macro = f1_score(tweets_y, clf.predict(tfidf_test), average="macro")
        results_processed = generate_results(tweets_y, clf.predict(tfidf_test), classes, folder)
        print acc
        print f1_macro
        results_to_visualize = []
        for tweet, code_1 in zip(tweets, clf.predict(tfidf_test)):
            tweet_visualize = {}
            tweet_visualize['cleaned_text'] = tweet['cleaned_text']
            tweet_visualize['truth'] = tweet[config['class']]
            tweet_visualize['url'] = tweet['url']
            tweet_visualize['code_1'] = code_1
            results_to_visualize.append(tweet_visualize)
        json.dump(results_to_visualize, open(folder + "results-visualize-annotated-tweets-svm.json", "w"))

def start_training(config=None, folder=None, classes=None):
    if config['architecture'] == "classic":
        classic(config, folder, classes)
    elif config['architecture'] == "two_part_classifier":
        two_part_classifier(config)

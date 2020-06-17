import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from drawing_utils import ConfusionMatrixDrawer
from constants import FOLDS_NUMBER
import time, json, os, glob
from data_preprocessing import get_classes_from_target_class, find_1
from keras.models import load_model
from keras import backend as K
from imblearn.metrics import geometric_mean_score 
from custom_classes import L2Norm
from keras.utils import CustomObjectScope
from sklearn.preprocessing import MultiLabelBinarizer

ACCURACY_AT = 3


def get_multi_labels_from_onehot(classes, multi_labels):
    new_labels = []
    for labels in multi_labels:
        new_multilabel = []
        for i,label in enumerate(labels):
            if label == 1.0:
                new_multilabel.append(classes[i])
        new_labels.append(new_multilabel)
    return new_labels
def generate_results(raw_test_y, raw_predicted_y, classes, folder):
    print "Unique values in raw_test_y: " + str(np.unique(np.array(raw_test_y)))
    print "Unique values raw_predicted_y: " + str(np.unique(raw_predicted_y))
    conf_matrix = confusion_matrix(raw_test_y, raw_predicted_y, labels=classes)
    if folder != None:
        ConfusionMatrixDrawer(conf_matrix, classes, folder, True, True, title='Confusion matrix, without normalization')
        ConfusionMatrixDrawer(conf_matrix, classes, folder, True, True, normalize=True, title='Normalized confusion matrix')
    print conf_matrix
    print raw_test_y[0]
    print raw_predicted_y[0]
    acc = accuracy_score(raw_test_y, raw_predicted_y)
    print "The accuracy of the model using scikit is: " + str(acc)
    f1_macro = f1_score(raw_test_y, raw_predicted_y, average="macro")
    recall_macro = recall_score(raw_test_y, raw_predicted_y, average="macro")
    precision_macro = precision_score(raw_test_y, raw_predicted_y, average="macro")
    f1_micro = f1_score(raw_test_y, raw_predicted_y, average="micro")
    recall_micro = recall_score(raw_test_y, raw_predicted_y, average="micro")
    precision_micro = precision_score(raw_test_y, raw_predicted_y, average="micro")
    print classes
    geometric = geometric_mean_score(raw_test_y, raw_predicted_y, labels=classes, average="macro")
    geometric_2 =geometric_mean_score(raw_test_y, raw_predicted_y, labels=classes, correction=0.001)
    print (classification_report(raw_test_y, raw_predicted_y))
    results = {}
    results['confusion_matrix'] = conf_matrix.tolist()
    results['precision_macro'] = precision_macro
    results['recall_macro'] = recall_macro
    results['f1_macro'] = f1_macro
    results['precision_micro'] = precision_micro
    results['recall_micro'] = recall_micro
    results['f1_micro'] = f1_micro
    results['acc'] = acc
    results['geometric'] = geometric
    results['geometric_2'] = geometric_2
    print "Geometric:" + str(geometric)
    print "Geometric 2:" + str(geometric_2)
    results['classes'] =  {}
    for c in classes:
        results['classes'][c] = evaluation_per_class(raw_test_y, raw_predicted_y, c)
    results_processed = [results['acc'],results['f1_macro'],results['recall_macro'], results['precision_macro'], results['geometric'], results['geometric_2']]
    for element in classes:
        results_processed.append(results['classes'][element]['f1'])
    return results_processed


def evaluate_metric_at(config, predictions, y_labels, accuracy_at):
    softmax_values = []
    predicted_classes = []
    y_labels = y_labels[0]
    correct = [0] * accuracy_at
    prediction_range = accuracy_at
    for i, prediction in enumerate(predictions):
        correct_answer = y_labels[i].index(1)
        best_n = np.sort(prediction)[::-1][:prediction_range]
        softmax_values.append([])
        predicted_classes.append([])
        for j in range(prediction_range):
            softmax_values[i].append(best_n[j])
            predicted_classes[i].append(get_classes_from_target_class(config['class'])[prediction.tolist().index(
                best_n[j])])
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j, prediction_range):
                    correct[k] += 1

    accuracies = {}
    for i in range(prediction_range):
        accuracy = (correct[i] * 1.0) / len(y_labels)
        print '%s prediction accuracy: %s' % (i + 1, accuracy)
        accuracies['acc_' + str(i + 1)] = (accuracy)
    return accuracies, softmax_values, predicted_classes

def evaluation_per_class(true_y, predicted_y, class_to_train):
    results = {}
    b_true_y = to_binary_labels(true_y, class_to_train)
    b_predicted_y = to_binary_labels(predicted_y, class_to_train)
    acc = accuracy_score(b_true_y, b_predicted_y)
    f1 = f1_score(b_true_y, b_predicted_y)
    recall = recall_score(b_true_y, b_predicted_y)
    precision = precision_score(b_true_y, b_predicted_y)
    results['acc'] = acc
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    return results

def to_binary_labels(y_labels, class_to_train):
    binary_y_labels = []
    for label in y_labels:
        new_label = 0
        if label ==  class_to_train:
            new_label = 1
        binary_y_labels.append(new_label)
    return binary_y_labels

def predict_model(config, model, test_X, test_y, previous_phrase,  previous_previous, post_phrase, party, classes_2=None):
    x_dict = {}
    y_dict = {}
    if not previous_phrase and not party and not previous_previous and not post_phrase:
        x_dict = {"phrases": np.array(test_X[0])}
    elif previous_phrase and not party and not previous_previous and not post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1])}
    elif previous_phrase and not party and not previous_previous and  post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1]), "post_phrases": np.array(test_X[2]) }
    elif previous_phrase and party and not previous_previous and  post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1]), "post_phrases": np.array(test_X[2]), "party": np.array(test_X[3]) }
    elif previous_phrase and not party and previous_previous and not post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1]), "previous_previous_phrases": np.array(test_X[2])}
    elif not previous_phrase and party and not previous_previous and not post_phrase:
        """print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print "meto los partiesdaishdajhdjkashdsjadhaskdhasdkjasd"
        print test_X[1]"""
        x_dict = {"phrases": np.array(test_X[0]), "party": np.array(test_X[1])}
    elif previous_phrase and party and not previous_previous and not post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1]), "party": np.array(test_X[2])}
    elif previous_phrase and party and previous_previous and not post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "previous_phrases": np.array(test_X[1]), "previous_previous_phrases": np.array(test_X[2]), "party": np.array(test_X[3])}        
    elif not previous_phrase and not party and not previous_previous and post_phrase:
        x_dict = {"phrases": np.array(test_X[0]), "post_phrases": np.array(test_X[1])}
    if classes_2 != None and not config['to_freeze']:
        print "lelelelellelelelellelelelellelelelellelelelellelelelellelelelellelelelel"
        y_dict = {"main": np.array(test_y[0]), "secondary": np.array(test_y[1])}
    elif config['to_freeze'] and not config['architecture'] == 'adrian_architecture':
        y_dict = {"main": np.array(test_y[1])}
    elif (config['architecture'] == 'adrian_architecture' or config['architecture'] == 'gusy_architecture') and not config['to_freeze']:
        y_dict = {"main_subdomain": np.array(test_y[0]), "main":np.array(test_y[1])}
    else:
        y_dict = {"main": np.array(test_y[0])}

    if config['architecture'] == 'composed_loss' and "freezed_model" not in config:
        x_dict['gt_domain'] = np.array(test_y[0])
        x_dict['gt_subdomain'] = np.array(test_y[1])
    elif config['architecture'] == 'composed_loss' and "freezed_model" in config:
        x_dict['gt_domain'] = np.array(test_y[1])
        x_dict['gt_subdomain'] = np.array(test_y[0])     
    predicted_y = model.predict(x_dict)
    metrics = model.evaluate(x_dict, y_dict)
    return predicted_y, metrics
    #metrics = model.evaluate({"phrases": np.array(test_X[0])},np.array(test_y))
    #predicted_y = model.predict({"phrases": np.array(test_X[0])})
    #return predicted_y


def process_foldings(all_folds_results):
    results = [.0] * len(all_folds_results[0])
    for i in range(len(all_folds_results)):
        for j in range(len(all_folds_results[0])):
            results[j] += all_folds_results[i][j]
    for i in range(len(all_folds_results[0])):
        results[i] /= float(len(all_folds_results))
    return results

#False: domain to subdomain. True: subdomain to domain.
def softmax_prioritizer(config, domain_softmax, domain_classes, subdomain_softmax, subdomain_classes, mode=False):
    prioritizer = []
    prediction_range = 2
    predicted_classes_domain = []
    predicted_classes_subdomain = []
    for i,(e_domain, e_subdomain) in enumerate(zip(domain_softmax, subdomain_softmax)):
        best_n = np.sort(e_domain)[::-1][:prediction_range]
        best_n_sub = np.sort(e_subdomain)[::-1][:prediction_range]
        predicted_classes_domain.append([])
        predicted_classes_subdomain.append([])
        for j in range(prediction_range):
            #softmax_values[i].append(best_n[j])
            predicted_classes_domain[i].append(get_classes_from_target_class(config['class_2'])[e_domain.tolist().index(
                best_n[j])])
            predicted_classes_subdomain[i].append(get_classes_from_target_class(config['class'])[e_subdomain.tolist().index(
                best_n_sub[j])])
        #print predicted_classes_domain[i]
        #print predicted_classes_subdomain[i]
        if predicted_classes_domain[i][0][0] == predicted_classes_subdomain[i][0][0]:
            prioritizer.append(predicted_classes_subdomain[i][0])
        elif predicted_classes_domain[i][0][0] != predicted_classes_subdomain[i][0][0] and predicted_classes_domain[i][0][0] == predicted_classes_subdomain[i][1][0]:
            prioritizer.append(predicted_classes_subdomain[i][1])
        elif predicted_classes_domain[i][0][0] != predicted_classes_subdomain[i][0][0] and predicted_classes_domain[i][0][0] != predicted_classes_subdomain[i][1][0] and predicted_classes_domain[i][1][0] != predicted_classes_subdomain[i][1][0]:
            prioritizer.append(predicted_classes_subdomain[i][1])
        else:
            prioritizer.append(predicted_classes_subdomain[i][0])

    return prioritizer

def evaluate_model(config, folder, test_X, test_y, model_folder=None, only_models=None, after_finetune=False):
    if config['to_freeze']:
        config['class'] = config['class_2']
    if model_folder == None:
        model_folder = folder
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    all_folds_results = []
    all_folds_results_subdomain = []
    test_y_labels = []
    test_y_labels_subdomain = []
    results = None
    results_subdomain = None
    if config['sequential']:
        for split in test_y[0]:
            #split_y = []
            for label in split:
                test_y_labels.append(get_classes_from_target_class(config['class'])[find_1(label)])
            #test_y_labels.append(split_y)
    else:
        if config['to_freeze']:
            for label in test_y[1]:
                test_y_labels.append(get_classes_from_target_class(config['class'])[find_1(label)])
        else:
            for label in test_y[0]:
                test_y_labels.append(get_classes_from_target_class(config['class'])[find_1(label)])
        if config['architecture'] == 'composed_loss':
            for label in test_y[1]:
                test_y_labels_subdomain.append(get_classes_from_target_class(config['class_2'])[find_1(label)])
    if config['cross_val_ready']:
        for i in range(FOLDS_NUMBER):
            if config['multi_embedding'] or config['multi_embedding_multi_file']:
                model = load_model(model_folder + "../best_model_" + str(i) + ".hdf5")
            else:
                if config['architecture'] == "composed_loss":
                    if only_models != None:
                        model = only_models[i]
                        model.load_weights(model_folder + "best_model_" + str(i) + ".hdf5")
                else:
                    if config['use_normalizer']:
                        with CustomObjectScope({'L2Norm': L2Norm}):
                            model = load_model(model_folder + "best_model_" + str(i) + ".hdf5")
                    else:
                        model = load_model(model_folder + "best_model_" + str(i) + ".hdf5")
            predicted_y, metrics = predict_model(config, model, test_X, test_y, config['previous_phrase'], config['previous_previous'], config['post_phrase'], config['party'])
            #predicted_y = probas_to_classes(predicted_y)
            class_d = None
            class_sub = None
            if config['architecture'] == 'composed_loss' and "freezed_model" not in config:
                print "primero"
                print "Shape 1: " + str(predicted_y[0].shape)
                print "Shape 2" + str(predicted_y[1].shape)
                predicted_y_subdomain = predicted_y[1].argmax(axis=-1)
                predicted_y = predicted_y[0].argmax(axis=-1)
                class_d = config['class']
                class_sub = config['class_2']
            elif config['architecture'] == 'composed_loss' and "freezed_model" in config:
                print "segundo"
                print "Shape 1: " + str(predicted_y[0].shape)
                print "Shape 2" + str(predicted_y[1].shape)
                predicted_y_subdomain = predicted_y[1].argmax(axis=-1)
                predicted_y = predicted_y[0].argmax(axis=-1) 
                class_d = config['class_2']
                class_sub = config['class']
            elif config['architecture'] == 'adrian_architecture':
                accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, predicted_y[1], test_y, ACCURACY_AT)
                predicted_y = predicted_y[1].argmax(axis=-1)
            elif config['architecture'] == 'gusy_architecture':
                accuracies_at_sub, softmax_values_sub, predicted_classes_sub = evaluate_metric_at(config, predicted_y[1], test_y, ACCURACY_AT)
                predicted_y_sub = predicted_y[1].argmax(axis=-1)
                print str(len(predicted_y[1]))
                print str(len(predicted_y[1][0]))
                print str(len(predicted_y[0]))
                print str(len(predicted_y[0][0]))
                accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, predicted_y[0], [test_y[1]], ACCURACY_AT)
                predicted_y_dom = predicted_y[0].argmax(axis=-1)
                new_softmax_values = [] 
                for soft_values in predicted_y[1]:
                    new_softmax_values_per_e = []
                    for i, v in enumerate(soft_values):
                        the_classes = get_classes_from_target_class(config['class'])
                        the_other_classes = get_classes_from_target_class(config['class_2'])
                        tmp_class = the_classes[i]
                        go_int = int(tmp_class[0])
                        if go_int == 0:
                            new_softmax_values_per_e.append(v)
                        else:
                            go_int = go_int -1
                            new_softmax_values_per_e.append(v * predicted_y[0][i][go_int])
                    new_softmax_values.append(np.asarray(new_softmax_values_per_e))
                print str(len(new_softmax_values))
                print str(len(new_softmax_values[0]))
                print str(new_softmax_values[0])
                accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, np.asarray(new_softmax_values), [test_y[0]], ACCURACY_AT)
                print accuracies_at
                prioritizer = softmax_prioritizer(config, predicted_y[0], get_classes_from_target_class(config['class_2']), predicted_y[1], get_classes_from_target_class(config['class']))
                tmp_test_y = []
                for e in test_y[0]:
                    tmp_test_y.append(get_classes_from_target_class(config['class'])[find_1(e)])
                print tmp_test_y[0]
                print prioritizer[0]
                acc = accuracy_score(tmp_test_y, prioritizer)
                print "The accuracy of the model using scikit is: " + str(acc)
                predicted_y = np.asarray(new_softmax_values).argmax(axis=-1)
            else:
                accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, predicted_y, test_y, ACCURACY_AT)
                predicted_y = predicted_y.argmax(axis=-1)
            raw_predicted_y = []
            raw_predicted_y_subdomain = []
            for label in predicted_y:
                class_d = config['class']
                raw_predicted_y.append(get_classes_from_target_class(class_d)[label])
            if config['architecture'] == 'composed_loss':
                for label in predicted_y_subdomain:
                    raw_predicted_y_subdomain.append(get_classes_from_target_class(class_sub)[label])
            if not (config['architecture'] == 'composed_loss' and "freezed_model" in config):
                tmp_results = generate_results(test_y_labels, raw_predicted_y, get_classes_from_target_class(class_d), folder)
                results = [tmp_results[0], accuracies_at['acc_1'], accuracies_at['acc_2'], accuracies_at['acc_3']] + tmp_results[1:]       
                all_folds_results.append(results)
                #all_folds_results_subdomain.append(generate_results(test_y_labels_subdomain, raw_predicted_y_subdomain, get_classes_from_target_class(class_sub), folder))
            else:
                all_folds_results_subdomain.append(generate_results(test_y_labels, raw_predicted_y_subdomain, get_classes_from_target_class(class_sub), folder))
            #K.clear_session()
        print all_folds_results
        if not (config['architecture'] == 'composed_loss' and "freezed_model" in config):
            results = process_foldings(all_folds_results)
        if config['architecture'] == 'composed_loss':
            results = []
            results_subdomain = process_foldings(all_folds_results_subdomain)
    elif not config['sequential']:
        if config['use_normalizer']:
            with CustomObjectScope({'L2Norm': L2Norm}):
                model = load_model(model_folder + "best_model.hdf5")
        else:
            model = load_model(model_folder + "best_model.hdf5")
        predicted_y, metrics = predict_model(config, model, test_X, test_y, config['previous_phrase'], config['previous_previous'], config['post_phrase'], config['party'])
        if config['architecture'] == 'adrian_architecture' or config['architecture'] == 'gusy_architecture':
            accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, predicted_y[0], [test_y[1]], ACCURACY_AT)
            accuracies_at_subdomain, softmax_values_subdomain, predicted_classes_subdomain = evaluate_metric_at(config, predicted_y[1], [test_y[0]], ACCURACY_AT)
            softmax_prioritizer(config, softmax_values, get_classes_from_target_class(config['class_2']), softmax_values_subdomain, get_classes_from_target_class(config['class']))
            predicted_y = predicted_y[1].argmax(axis=-1)            
        elif config['architecture'] == 'multi_label' and after_finetune:
            print "FInetuning multilabel"
            print predicted_y[0]
            print metrics
            predicted_y[predicted_y>=0.5] = 1
            predicted_y[predicted_y<0.5] = 0
            predicted_y = get_multi_labels_from_onehot(get_classes_from_target_class(config['class']), predicted_y)
            test_y_labelized = get_multi_labels_from_onehot(get_classes_from_target_class(config['class']), test_y[0])
            print accuracy_score(MultiLabelBinarizer().fit_transform(test_y_labelized), MultiLabelBinarizer().fit_transform(predicted_y))
            """print predicted_y[0]
            print test_y[0][0]
            print type(predicted_y[0])
            print type(test_y[0][0])
            print f1_score(test_y[0], predicted_y, average='macro')
            print accuracy_score(test_y[0], predicted_y)"""
        else:
            accuracies_at, softmax_values, predicted_classes = evaluate_metric_at(config, predicted_y, test_y, ACCURACY_AT)
            predicted_y = predicted_y.argmax(axis=-1)
        raw_predicted_y = []
        for label in predicted_y:
            raw_predicted_y.append(get_classes_from_target_class(config['class'])[label])
        results = generate_results(test_y_labels, raw_predicted_y, get_classes_from_target_class(config['class']), folder)
        results = [results[0], accuracies_at['acc_1'], accuracies_at['acc_2'], accuracies_at['acc_3']] + results[1:]
        save_filename_depending = 'results-'
        if after_finetune:
            save_filename_depending = 'twitter-' + save_filename_depending
        json.dump(results, open(folder + save_filename_depending + timestamp + ".json", "w"))

        return softmax_values, predicted_classes
    else:
        model = load_model(model_folder + "best_model.hdf5")
        predicted_y, metrics = predict_model(config, model, test_X, test_y, False, False, False, False)
        raw_predicted_y = []
        for split in predicted_y:
            split_argmax = split.argmax(axis=-1)
            for y in split_argmax:
                raw_predicted_y.append(get_classes_from_target_class(config['class'])[y])
        results = generate_results(test_y_labels, raw_predicted_y, get_classes_from_target_class(config['class']), folder)
    print folder
    json.dump(results, open(folder + "results-" + timestamp + ".json", "w"))
    if config['architecture'] == 'composed_loss':
        json.dump(results_subdomain, open(folder + "results-subdomain" + timestamp + ".json", "w"))
    #return softmax_values, predicted_classes

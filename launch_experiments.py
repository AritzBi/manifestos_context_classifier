from comet_ml import Experiment
import argparse
from data_preprocessing import generate_sets,get_classes_from_target_class, find_1, generate_sequential_sets, compute_weight_classes
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
import time, glob, os, json
from keras import backend as K
from evaluation import generate_results, predict_model, evaluate_model
from sequential_approach import create_lstm_sequential_model, create_lstm_for_sequences, create_lstm_cnn_for_sequences
from cnn_approach import create_model_late_fusion, generate_previous_phrase_as_attention, create_model_channel, custom_loss_experiment, custom_loss_experiment_domain_freezed, multi_stream_single_loss, adrian_architecture, multi_label_architecture, replace_sigmoid_with_softmax
import numpy as np
from classify_tweets import classify_annotated_tweets, predict_tweets, handle_annotated_english_tweets, get_english_annotated_tweets_sets, classify_non_annotated_tweets
from scikit_approaches import start_training
from soft_attention import gru_attention_model

def train_model(config,train_X, train_y, eval_X, eval_y, folder, fold=None, multilingua=None, only_models = False, finetune=None ):
    train_x_dict = {"phrases": np.array(train_X[0])}
    eval_x_dict = {"phrases": np.array(eval_X[0])}
    if config['previous_phrase']:
        train_x_dict['previous_phrases'] = np.array(train_X[1])
        eval_x_dict['previous_phrases'] = np.array(eval_X[1])
        if config['party'] and (config['party_as_one_hot'] or config['party_as_rile_score'] or config[
        'party_as_std_mean'] or config['party_as_deconv']) and not config['post_phrase']:
            train_x_dict['party'] = np.array(train_X[2])
            eval_x_dict['party'] = np.array(eval_X[2])
        if config['post_phrase']:
            train_x_dict['post_phrases'] = np.array(train_X[2])
            eval_x_dict['post_phrases'] = np.array(eval_X[2])
        if config['party'] and (config['party_as_one_hot'] or config['party_as_rile_score'] or config[
        'party_as_std_mean'] or config['party_as_deconv']) and config['post_phrase']:
            train_x_dict['party'] = np.array(train_X[3])
            eval_x_dict['party'] = np.array(eval_X[3])     
    elif config['party'] and (config['party_as_one_hot'] or config['party_as_rile_score'] or config[
        'party_as_std_mean'] or config['party_as_deconv']):
        train_x_dict['party'] = np.array(train_X[1])
        eval_x_dict['party'] = np.array(eval_X[1])
    if config['previous_previous']:
        train_x_dict['previous_previous_phrases'] = np.array(train_X[2])
        print "a ver que pasa crack"  + str(len(train_X[2]))
        eval_x_dict['previous_previous_phrases'] = np.array(eval_X[2])
    if config['post_phrase']:
        train_x_dict['post_phrases'] = np.array(train_X[1])
        eval_x_dict['post_phrases'] = np.array(eval_X[1])
    if finetune != None:
        model = finetune        
    elif config['architecture'] == "lstm":
        model = create_lstm_sequential_model(config, len(get_classes_from_target_class(config['class'])), folder)
    elif config['architecture'] == "cnn":
        if config['to_freeze']:
            model = create_model_late_fusion(config, len(get_classes_from_target_class(config['class_2'])), folder, multilingua)
        else:
            model = create_model_late_fusion(config, len(get_classes_from_target_class(config['class'])), folder, multilingua)
    elif config['architecture'] == "cnn_channel":
        model = create_model_channel(config, len(get_classes_from_target_class(config['class'])), folder, multilingua)
    elif config['architecture'] == "sequential_lstm":
        model = create_lstm_cnn_for_sequences(config, len(get_classes_from_target_class(config['class'])), folder)
    elif config['architecture'] == "gru_attention":
        model = gru_attention_model(config, len(get_classes_from_target_class(config['class'])), folder)
    elif config['architecture'] == "prev_phrase_attention":
        model = generate_previous_phrase_as_attention(config, len(get_classes_from_target_class(config['class'])), folder)
    elif config['architecture'] == "composed_loss" and 'freezed_model' not in config:
        model = custom_loss_experiment(config, len(get_classes_from_target_class(config['class'])), len(get_classes_from_target_class(config['class_2'])), folder, multilingua)
    elif config['architecture'] == "composed_loss" and config['freezed_model']:
        model = custom_loss_experiment_domain_freezed(config, len(get_classes_from_target_class(config['class'])), len(get_classes_from_target_class(config['class_2'])), folder, multilingua)
    elif config['architecture'] == "multi_stream_single_loss" and config['freezed_model']:
        model =  multi_stream_single_loss(config, len(get_classes_from_target_class(config['class'])), folder, multilingua)
    elif config['architecture'] == "adrian_architecture" or config['architecture'] == 'gusy_architecture':
        model = adrian_architecture(config, len(get_classes_from_target_class(config['class_2'])), len(get_classes_from_target_class(config['class'])), folder, None)
    elif config['architecture'] == 'multi_label':
        model = multi_label_architecture(config, len(get_classes_from_target_class(config['class'])), folder)
    if fold == None:
        model_name = "best_model.hdf5"
    else:
        model_name = "best_model_" + str(fold) + ".hdf5"
    class_weight_2 = None
    if config['weight_class']:
        class_weight_2 = compute_weight_classes(get_classes_from_target_class(config['class']), train_y[0])
        print "CLASS WEIGHTING"
        print class_weight_2
    print "Save model folder" + folder
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=config['patience'], verbose=0, mode='auto')
    #tensorboard = TensorBoard(log_dir="./my_log_dir", histogram_freq=0)
    #embeddings_layer_names = ['embedding_1']

    #tb_callback = TensorBoard(histogram_freq=10, write_graph=False,
                              #embeddings_freq=100,
                              #embeddings_layer_names=embeddings_layer_names)
    """experiment = Experiment(api_key="1U6LMYHksUq1mhuRyGi0hU9bG", project_name="general", workspace="aritzbi")
    experiment.log_multiple_params(config)"""

    if not config['architecture'] == 'composed_loss' and not only_models:
        if config['architecture'] == 'multi_label' and finetune:
            checkpoint = ModelCheckpoint(folder + model_name, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
            print "ES CATEGORICAL ACCURACY"
        else:
            checkpoint = ModelCheckpoint(folder + model_name, monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        if config['to_freeze']:
            model.fit(train_x_dict , np.array(train_y[1]), validation_data=(eval_x_dict, np.array(eval_y[1])), epochs=config['number_epochs'], batch_size=config['batch_size'], callbacks=[checkpoint, early_stopping], class_weight=class_weight_2)
        elif config['architecture'] == "adrian_architecture" or config['architecture'] == 'gusy_architecture':
            checkpoint = ModelCheckpoint(folder + model_name, monitor="val_main_subdomain_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
            model.fit(train_x_dict , {"main_subdomain": np.array(train_y[0]), "main":np.array(train_y[1])}, validation_data=(eval_x_dict, {"main_subdomain":np.array(eval_y[0]), "main": np.array(eval_y[1])}), epochs=config['number_epochs'], batch_size=config['batch_size'], callbacks=[checkpoint, early_stopping], class_weight=class_weight_2)
        else:
            model.fit(train_x_dict , np.array(train_y[0]), validation_data=(eval_x_dict, np.array(eval_y[0])), epochs=config['number_epochs'], batch_size=config['batch_size'], callbacks=[checkpoint, early_stopping], class_weight=class_weight_2)
    else:
        if not config['architecture'] == 'composed_loss' and only_models:
            checkpoint = ModelCheckpoint(folder + model_name, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
        else:
            checkpoint = ModelCheckpoint(folder + model_name, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
            model.compile(loss = None, optimizer = 'adam') 
        if only_models:
            return model
        model.summary()
        train_x_dict['gt_domain'] = np.array(train_y[1])
        train_x_dict['gt_subdomain'] = np.array(train_y[0])
        eval_x_dict['gt_domain'] = np.array(eval_y[1])
        eval_x_dict['gt_subdomain'] = np.array(eval_y[0])
        model.fit(x=train_x_dict, validation_data=(eval_x_dict,None), epochs=config[
        'number_epochs'], batch_size=config['batch_size'], callbacks=[checkpoint, early_stopping],
              class_weight=class_weight_2)
    return model

def config_defaults(config):
    if "party_as_one_hot" not in config:
        config["party_as_one_hot"] = False
    if "dropout_before_party" not in config:
        config["dropout_before_party"] =  False
    if "post_phrase" not in config:
        config["post_phrase"] = False
    if "previous_previous" not in config:
        config['previous_previous'] = False
    if "sequential" not in config:
        config["sequential"] = False
    if "dataset_folder" not in config:
        config['dataset_folder'] = "./datasets/rmp_dataset/2_tokenized/"
    if "sequential_data" not in config:
        config['sequential_data'] = False
    if "add_annotated_tweets" not in config:
        config['add_annotated_tweets'] = False
    if "sci-kit" not in config:
        config['sci-kit'] = False
    if "class_2" not in config:
        config['class_2'] =  None
    if "no_padding_for_lstms" not in config:
        config['no_padding_for_lstms'] =  False
    if "add_non_annotated_tweets" not in config:
        config['add_non_annotated_tweets'] = False
    if "attention" not in config:
        config['attention'] = False
    if "max_words" not in config:
        config['max_words'] = False
    if "hard_attention" not in config:
        config['hard_attention'] = False
    if "seed" not in config:
        config['seed'] = 0
    if "multilingual" not in config:
        config['multilingual'] = False
    if "party_as_rile_score" not in config:
        config['party_as_rile_score'] = False
    if "party_as_std_mean" not in config:
        config['party_as_std_mean'] = False
    if "language" not in config:
        config['language'] = False
    if "multi_embedding" not in config:
        config["multi_embedding"] = False
    if "multi_embedding_multi_file" not in config:
        config["multi_embedding_multi_file"] = False
    if "embedding_with_manifesto" not in config:
        config["embedding_with_manifesto"] = False
    if "one_language" not in config:
        config['one_language'] = True
    if "one_language_select" not in config:
        config['one_language_select'] = False
    if "topfish" not in config:
        config["topfish"] = False
    if "languages" not in config:
        config["languages"] = ['german', 'danish', 'spanish', 'italian', 'english', 'finnish', 'french']
    if "party_as_deconv" not in config:
        config["party_as_deconv"] = False
    if "to_freeze" not in config:
        config['to_freeze'] = False
    if "concatenate_cnn_features" not in config:
        config['concatenate_cnn_features'] = False
    if "dense_after_feature_concatenation" not in config:
        config["dense_after_feature_concatenation"] =  False
    if "share_embedding" not in config:
        config['share_embedding'] = False
    if "feifei" not in config:
        config['feifei'] = None
    if "use_normalizer" not in config:
        config['use_normalizer'] = False
    if "add_english_annotated_tweets" not in config:
        config['add_english_annotated_tweets'] = False
    if "finetune" not in config:
        config['finetune'] = False
    if "big_test" not in config:
        config['big_test'] = False
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some options.')
    parser.add_argument("-s", "--specific",action="store_true", dest="specifc",default=False,help="To execute specific experiments")
    parser.add_argument("-ats", "--a_tweets_spanish", action="store_true", dest="a_tweet", default= False)
    parser.add_argument("-ate", "--a_tweets_english", action="store_true", dest="a_tweet_english", default= False)
    parser.add_argument("-nte", "--non-tweets_english", action="store_true", dest="non_tweets_english", default=False)
    parser.add_argument("-t", "--tweets", action="store_true", dest="tweet", default= False)
    parser.add_argument("-gpu", "--gpu",action="store", dest="gpu", default="0", help="To choose the GPU")
    parser.add_argument("-e", "--evaluate", action="store_true", dest="e", default=False)
    parser.add_argument("-mlingual", "--multilingual", action="store_true", dest="multilingual", default=False)
    parser.add_argument("-ft", "--fasttext", action="store_true", dest="fasttext", default=False)
    parser.add_argument("-nz", "--nozero", action="store_true", dest="nozero", default=False)
    parser.add_argument("-me", "--multiembedding", action="store_true", dest="multiembedding", default=False)
    parser.add_argument("-memf", "--multiembeddingmultifile", action="store_true", dest="multiembeddingmultifile", default=False)
    parser.add_argument("-topfish", "--topfish_emb", action="store_true", dest="topfish_emb", default=False)
    parser.add_argument("-topfishlang", "--topfish_language", action="store_true", dest="topfishlang", default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    #folder = './datasets/rmp_dataset/elder_experiments/'
    folder = './datasets/english_annotated_tweets/experiments/'
    print str(args.e)
    if args.topfishlang:
        languages = ['german', 'italian', 'english', 'french']
    else:
        #languages = ['german', 'danish', 'spanish', 'italian', 'english', 'finnish', 'french']
        languages = ['english']
    if not args.specifc:
        if args.gpu == '0':
            folder += 'domain/'
        elif args.gpu == '1':
            folder += 'subdomain/'
        filelist = glob.glob(os.path.join(folder, '*'))
    else:
        if args.gpu == '0':
            print "Domains"
            #experiments = ['baseline_2_domain','baseline_3_domain', 'baseline_2_domain_party', 'baseline_3_domain_party', 'baseline_2_domain_prev','baseline_3_domain_prev']
            #experiments = [ 'baseline_2_onlytweets', 'baseline_3_onlytweets', 'baseline_2_onlytweets_prev', 'baseline_3_onlytweets_prev', 'baseline_2_onlytweets_party', 'baseline_3_onlytweets_party']
            #experiments = ['baseline_2_domain_finetune', 'baseline_3_domain_finetune','baseline_2_domain_pre_finetune','baseline_3_domain_pre_finetune', 'baseline_2_domain_party_finetune', 'baseline_3_domain_party_finetune']
            experiments = ['baseline_2_domain_party_finetune']
            #E2
            #experiments = ['baseline_1_onlytweets', 'baseline_2_onlytweets', 'baseline_3_onlytweets', 'baseline_4_onlytweets', 'baseline_1_onlytweets_prev', 'baseline_2_onlytweets_prev', 'baseline_3_onlytweets_prev', 'baseline_4_onlytweets_prev', 'baseline_1_onlytweets_party', 'baseline_2_onlytweets_party', 'baseline_3_onlytweets_party', 'baseline_4_onlytweets_party', 'baseline_1_onlytweets_bigtest', 'baseline_2_onlytweets_bigtest', 'baseline_3_onlytweets_bigtest', 'baseline_4_onlytweets_bigtest']
            #E3
            #experiments = ['baseline_1_domain_pre_finetune','baseline_2_domain_pre_finetune','baseline_3_domain_pre_finetune','baseline_4_domain_pre_finetune',]
            #experiments = ['baseline_1_onlytweets_bigtest', 'baseline_2_onlytweets_bigtest', 'baseline_3_onlytweets_bigtest', 'baseline_4_onlytweets_bigtest']
            #experiments = ['baseline_1_domain_party_finetune', 'baseline_2_domain_party_finetune', 'baseline_3_domain_party_finetune', 'baseline_4_domain_party_finetune']
            #experiments = ['baseline_domain_bigtest_party','baseline_2_domain_bigtest_party','baseline_3_domain_bigtest_party','baseline_4_domain_bigtest_party']
            #experiments = ['baseline_domain','baseline_2_domain','baseline_3_domain','baseline_4_domain','baseline_1_domain_party', 'baseline_2_domain_party', 'baseline_3_domain_party', 'baseline_4_domain_party']
            #experiments = ['baseline_domain_bigtest','baseline_2_domain_bigtest','baseline_3_domain_bigtest','baseline_4_domain_bigtest','baseline_1_domain_finetune_bigtest', 'baseline_2_domain_finetune_bigtest', 'baseline_3_domain_finetune_bigtest', 'baseline_4_domain_finetune_bigtest']
            #experiments = ['baseline_1_onlytweets', 'baseline_2_onlytweets', 'baseline_3_onlytweets', 'baseline_4_onlytweets']
            #experiments = ['baseline_1_domain_finetune', 'baseline_2_domain_finetune', 'baseline_3_domain_finetune', 'baseline_4_domain_finetune']
            #experiments = ['baseline_2_domain_bigtest', 'baseline_3_domain_bigtest', 'baseline_4_domain_bigtest', 'baseline_1_domain_finetune_bigtest', 'baseline_2_domain_finetune_bigtest', 'baseline_3_domain_finetune_bigtest', 'baseline_4_domain_finetune_bigtest']
            #experiments = ['baseline_domain','baseline_2_domain','baseline_3_domain','baseline_4_domain','baseline_ft_online_1_domain', 'baseline_ft_online_2_domain', 'baseline_ft_online_3_domain', 'baseline_ft_online_4_domain']
        elif args.gpu == '1':
            print "SubDomains"
            experiments = ['subdomain/baseline_2_subdomain_party_finetune']
            #experiments = ['subdomain/baseline_2_subdomain','subdomain/baseline_3_subdomain', 'subdomain/baseline_2_subdomain_prev','subdomain/baseline_3_subdomain_prev', 'subdomain/baseline_2_subdomain_party','subdomain/baseline_3_subdomain_party' ]
            #experiments = [ 'subdomain/baseline_2_subdomain_onlytweets', 'subdomain/baseline_3_subdomain_onlytweets','subdomain/baseline_2_subdomain_onlytweets_prev', 'subdomain/baseline_3_subdomain_onlytweets_prev','subdomain/baseline_2_subdomain_onlytweets_party', 'subdomain/baseline_3_subdomain_onlytweets_party']
            #experiments = ['subdomain/baseline_2_subdomain_finetune', 'subdomain/baseline_3_subdomain_finetune','subdomain/baseline_2_subdomain_party_finetune','subdomain/baseline_3_subdomain_party_finetune','subdomain/baseline_2_subdomain_prev_finetune','subdomain/baseline_3_subdomain_prev_finetune']
            #E1
            #experiments = ['subdomain/baseline_subdomain','subdomain/baseline_2_subdomain','subdomain/baseline_3_subdomain','subdomain/baseline_4_subdomain','subdomain/baseline_1_subdomain_prev', 'subdomain/baseline_2_subdomain_prev','subdomain/baseline_3_subdomain_prev','subdomain/baseline_4_subdomain_prev', 'subdomain/baseline_1_subdomain_party','subdomain/baseline_2_subdomain_party','subdomain/baseline_3_subdomain_party','subdomain/baseline_4_subdomain_party']
            #E2
            #experiments = ['subdomain/baseline_1_subdomain_onlytweets', 'subdomain/baseline_2_subdomain_onlytweets', 'subdomain/baseline_3_subdomain_onlytweets', 'subdomain/baseline_4_subdomain_onlytweets', 'subdomain/baseline_1_subdomain_onlytweets_prev', 'subdomain/baseline_2_subdomain_onlytweets_prev', 'subdomain/baseline_3_subdomain_onlytweets_prev', 'subdomain/baseline_4_subdomain_onlytweets_prev','subdomain/baseline_1_subdomain_onlytweets_party', 'subdomain/baseline_2_subdomain_onlytweets_party', 'subdomain/baseline_3_subdomain_onlytweets_party', 'subdomain/baseline_4_subdomain_onlytweets_party']
            #E3
            #experiments = ['subdomain/baseline_1_subdomain_finetune', 'subdomain/baseline_2_subdomain_finetune', 'subdomain/baseline_3_subdomain_finetune', 'subdomain/baseline_4_subdomain_finetune', 'subdomain/baseline_1_subdomain_party_finetune','subdomain/baseline_2_subdomain_party_finetune','subdomain/baseline_3_subdomain_party_finetune','subdomain/baseline_4_subdomain_party_finetune', 'subdomain/baseline_1_subdomain_prev_finetune','subdomain/baseline_2_subdomain_prev_finetune','subdomain/baseline_3_subdomain_prev_finetune','subdomain/baseline_4_subdomain_prev_finetune',]
            #experiments = ['subdomain/baseline_subdomain','subdomain/baseline_2_subdomain','subdomain/baseline_3_subdomain','subdomain/baseline_4_subdomain','subdomain/baseline_1_subdomain_party_finetune','subdomain/baseline_2_subdomain_party_finetune','subdomain/baseline_3_subdomain_party_finetune', 'subdomain/baseline_4_subdomain_party','subdomain/baseline_1_subdomain_party','subdomain/baseline_2_subdomain_party','subdomain/baseline_3_subdomain_party','subdomain/baseline_4_subdomain_party','subdomain/baseline_1_subdomain_finetune', 'subdomain/baseline_2_subdomain_finetune', 'subdomain/baseline_3_subdomain_finetune', 'subdomain/baseline_4_subdomain_finetune']
        filelist = [folder + s for s in experiments]
    for filenam in sorted(filelist):
        print "Running the experiment " + str(filenam.split('_')[-1])
        configs = []
        filenames = []
        #Multilingual: each language is trained and evaluated with its language data. 
        if args.multilingual and not args.multiembedding and not args.multiembeddingmultifile:
            print "MULTILINGUAL"
            for language in languages:
                config = json.load(open(filenam + '/config.json', 'r'))
                config = config_defaults(config)
                folder = filenam + "/../../" + language + "/"
                config['language'] = language
                config['embedding_size_1'] = 300
                if args.nozero:
                    if config['post_phrase']:
                        config['dataset'] = folder + "2_tokenized/" + language + "nozero_post.json"
                    else:
                        config['dataset'] = folder + "2_tokenized/" + language + "nozero.json"
                else:
                    if config['post_phrase']:
                        config['dataset'] = folder + "2_tokenized/" + language + "_post.json"
                    else:
                        config['dataset'] = folder + "2_tokenized/" + language + ".json"
                config['multilingual'] = True
                folder = filenam + "/" + language + "/"
                if args.fasttext:
                    config_ft = config.copy()
                    config_ft['embedding_type_1'] = 'fasttext'
                    if config['embedding_with_manifesto']:
                        config_ft['embedding_name_1'] = 'fasttext/with_manifestos/' + language + '.vec'
                    else:
                        config_ft['embedding_name_1'] = 'fasttext/without_manifestos/wiki.' + language + '.vec'
                    folder = filenam + "/" + language + "/fasttext/"
                    configs.append(config_ft)
                    filenames.append(folder)
                else:
                    if language == "spanish":
                        config['embedding_name_1'] = "model_4"
                    else:
                        config['embedding_name_1'] = language + ".bin"
                    configs.append(config)
                    filenames.append(folder)

        #Normal: normal execution with no multilanguage component. 
        elif not args.multilingual and not args.multiembedding and not args.multiembeddingmultifile:
            print "NORMAL"
            config = json.load(open(filenam + '/config.json', 'r'))
            config = config_defaults(config)
            filename = filenam + "/"
            if len(languages) > 0:
                config['language'] = 'english'
            configs.append(config)
            filenames.append(filename)
        #MULTIEMBEDDING: Single embedding for multiple languages.
        # nohup python -u -s launch_experiments.py -me -nz -topfish &
        elif not args.multilingual and args.multiembedding and not args.multiembeddingmultifile:
            print "MULTIEMBEDDING"
            for language in languages:
                config = json.load(open(filenam + '/config.json', 'r'))
                config = config_defaults(config)
                folder = filenam + "/../../" + language + "/"
                config['language'] = language
                if args.topfish_emb:
                    config['embedding_size_1'] = 300
                    config['embedding_name_1'] = "wiki.big-five.mapped.vec"
                    config['embedding_type_1'] = "other"
                else:
                    config['embedding_size_1'] = 511
                    config['embedding_name_1'] = "multilingua.txt"
                    config['embedding_type_1'] = "txt"
                config["multi_embedding"] = True
                #config['multilingual'] = True
                if args.nozero:
                    config['dataset'] = folder + "2_tokenized/" + language + "nozero.json"
                else:
                    config['dataset'] = folder + "2_tokenized/" + language + ".json"
                folder = filenam + "/" + language + "/"
                configs.append(config)
                filenames.append(folder)
        #MULTIEMBEDDING MULTIFILE: Multiple language embeddings each of them in their own vector space .
        elif not args.multilingual and not args.multiembedding and args.multiembeddingmultifile:
            for language in languages:
                config = json.load(open(filenam + '/config.json', 'r'))
                config = config_defaults(config)
                folder = filenam + "/../../" + language + "/"
                config['language'] = language
                config['embedding_size_1'] = 300
                if config['embedding_with_manifesto']:
                    if language == "english":
                        config['embedding_name_1'] = "fasttext/with_manifestos/" + language + ".vec"
                    else:
                        config['embedding_name_1'] = "fasttext/with_manifestos/align_" + language + ".vec"
                else:
                    if language == "english":
                        config['embedding_name_1'] = "fasttext/without_manifestos/wiki." + language + ".vec"
                    else:
                        config['embedding_name_1'] = "fasttext/without_manifestos/align.wiki." + language + ".vec"
                config['embedding_type_1'] = 'fasttext'
                config["multi_embedding_multi_file"] = True
                if args.nozero:
                    config['dataset'] = folder + "2_tokenized/" + language + "nozero.json"
                else:
                    config['dataset'] = folder + "2_tokenized/" + language + ".json"
                folder = filenam + "/" + language + "/"
                configs.append(config)
                filenames.append(folder)

        for config, folder in zip(configs, filenames):
            print config['embedding_type_1']
            print folder
        if (config['multi_embedding'] or config['multi_embedding_multi_file']) and not config['topfish']:
            multi_train_folds_X = [[[]],[[]],[[]],[[]],[[]]]
            multi_train_folds_y = [[[]],[[]],[[]],[[]],[[]]]
            multi_eval_fold_X = [[[]],[[]],[[]],[[]],[[]]]
            multi_eval_fold_y = [[[]],[[]],[[]],[[]],[[]]]
            multi_test_folds_X = {}
            multi_test_folds_y = {}
            for language in config['languages']:
                multi_test_folds_X[language] = {}
                multi_test_folds_y[language] = {}
            for config, filename_2 in zip(configs, filenames):
                if config['cross_val_ready']:
                    train_folds_X, o_train_folds, train_folds_y, eval_folds_X, o_eval_folds, eval_folds_y, o_test_phrases, test_X, test_y = generate_sets(
                        config, filename_2, multilingua=configs)
                    multi_test_folds_X[config['language']] = test_X
                    multi_test_folds_y[config['language']] = test_y
                    if not (config['one_language'] and config['one_language_select'] == config['language']):
                        for i, (train_X, train_y, eval_X, eval_y) in enumerate(
                            zip(train_folds_X, train_folds_y, eval_folds_X, eval_folds_y)):
                            print "Primer train_X " + str(len(train_X))
                            multi_train_folds_X[i][0] += train_X[0]
                            multi_train_folds_y[i][0] += train_y[0]
                            multi_eval_fold_X[i][0] += eval_X[0]
                            multi_eval_fold_y[i][0] += eval_y[0]
                    else:
                        print "NO ADDING ITALIAN"
            for i, (train_X, train_y, eval_X, eval_y) in enumerate(zip(multi_train_folds_X,multi_train_folds_y, multi_eval_fold_X, multi_eval_fold_y )):
                print "Segundo train_X " + str(len(train_X))
                train_model(config, train_X, train_y, eval_X, eval_y, filenam + "/", i, multilingua=configs)
            for config, filename_2 in zip(configs, filenames):
                evaluate_model(config, filename_2, multi_test_folds_X[config['language']], multi_test_folds_y[config['language']])
        else:
            models = []
            for config, filename_2 in zip(configs, filenames):
                if not config['sci-kit']:
                    if not args.a_tweet and not args.tweet and not args.a_tweet_english and not args.non_tweets_english:
                        if config['sequential']:
                            if config['cross_val_ready']:
                                pass
                            else:
                                o_train_splits, train_X, train_y, eval_X, eval_y, o_test_splits, test_X, test_y = generate_sequential_sets(config, filename_2)
                                train_model(config, train_X, train_y, eval_X, eval_y, filename_2)
                        else:
                            if config['cross_val_ready']:
                                train_folds_X, o_train_folds, train_folds_y, eval_folds_X,o_eval_folds, eval_folds_y, o_test_phrases, test_X, test_y = generate_sets(config,filename_2)
                                for i,(train_X, train_y, eval_X, eval_y) in enumerate(zip(train_folds_X, train_folds_y, eval_folds_X, eval_folds_y)):
                                    if not args.e:
                                        print "por aqui crack" + str(i)
                                        print config
                                        print filename_2
                                        models.append(train_model(config, train_X, train_y, eval_X, eval_y, filename_2, i))
                                        #if config['architecture'] == 'composed_loss':
                                            #models.append(train_model(config, train_X, train_y, eval_X, eval_y, filename_2, i, None, True))
                                        #K.clear_session()
                                    else:
                                        models.append(train_model(config, train_X, train_y, eval_X, eval_y, filename_2, i, None, True))   
                            else:
                                if config['sequential_data']:
                                    o_train_phrases, train_X, train_y, eval_X, eval_y, o_test_phrases, test_X, test_y = generate_sequential_sets(config, filename_2)
                                else:
                                    print "Paso por aqui cracks"
                                    if config['multi_embedding']:
                                        train_X, train_y, eval_X, eval_y, all_splits_languages_dict = generate_sets(config, filename_2, multilingua=configs)
                                    else:
                                        train_X, train_y, eval_X, eval_y, test_X, test_y,o_train_phrases, o_test_phrases, o_prev_test_phrases \
                                        = generate_sets(config, filename_2)
                                if not args.e:
                                    if config['multi_embedding']:
                                        train_model(config, train_X, train_y, eval_X, eval_y, filename_2, multilingua=configs)
                                    else:    
                                        train_model(config, train_X, train_y, eval_X, eval_y, filename_2)
                                K.clear_session()
                        if config['cross_val_ready']:
                            if config['architecture'] == 'composed_loss' or config['architecture'] == 'multi_stream_single_loss' :
                                evaluate_model(config, filename_2, test_X, test_y, None, models)
                            else:
                                evaluate_model(config, filename_2, test_X, test_y)
                        elif config['multi_embedding']:
                            for config, filename_2 in zip(configs, filenames):
                               evaluate_model(config, filename_2, [all_splits_languages_dict[config['language']]['test'][0]], [all_splits_languages_dict[config['language']]['test'][1]], filenames[0]) 

                        else:
                            softmax_values, predicted_classes = evaluate_model(config, filename_2, test_X, test_y)
                            results_to_visualize = []
                            for tok_phrase, tok_pev_phrase, ground_truth, softmax_value, predicted_class in zip(o_test_phrases,
                                                                                                o_prev_test_phrases, test_y[0],
                                                                                          softmax_values,
                                                                      predicted_classes):
                                phrase_data = {}
                                phrase_data['phrase'] = ' '.join(tok_phrase)
                                phrase_data['pev_phrase'] = ' '.join(tok_pev_phrase)
                                phrase_data['softmax_1'] = "{0:.2f}".format(softmax_value[0])
                                phrase_data['softmax_2'] = "{0:.2f}".format(softmax_value[1])
                                phrase_data['code_1'] = predicted_class[0]
                                phrase_data['code_2'] = predicted_class[1]
                                phrase_data['truth'] = get_classes_from_target_class(config['class'])[find_1(ground_truth)]
                                #print phrase_data
                                results_to_visualize.append(phrase_data)
                            json.dump(results_to_visualize, open(folder + "results-visualize.json", "w"))
                    else:
                        if not args.tweet:
                            if args.a_tweet_english:
                                config['language'] = 'english'
                                if config['finetune']:
                                    tweets_train_X, tweets_train_y, tweets_eval_X, tweets_eval_y, tweets_test_X, tweets_test_y = get_english_annotated_tweets_sets(config, folder)
                                    model = load_model(filename_2 + "best_model.hdf5")
                                    if config['architecture'] == 'multi_label':
                                        model = replace_sigmoid_with_softmax(config, get_classes_from_target_class(config['class']), filename_2, model )  
                                    print "is this none", model
                                    train_model(config, tweets_train_X, tweets_train_y, tweets_eval_X, tweets_eval_y, filename_2, None, None, False, model)
                                handle_annotated_english_tweets(config, filename_2)
                            else:
                                if args.non_tweets_english:
                                    classify_non_annotated_tweets(config, filename_2)
                                else:
                                    classify_annotated_tweets(config, filename_2)
                        else:
                            predict_tweets(config, filename_2)

                    K.clear_session()
                else:
                    start_training(config, filename_2, get_classes_from_target_class(config['class']))



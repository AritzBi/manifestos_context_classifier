from data_preprocessing import generate_sets, generate_embedding_matrix, get_classes_from_target_class, find_1
from keras.layers import Dense, LSTM, Input, Flatten, Reshape, concatenate, Dropout, Activation, BatchNormalization,\
    Conv1D, Conv2D, GlobalMaxPooling1D, GlobalMaxPooling2D, Concatenate, Bidirectional, GRU, TimeDistributed, Multiply, Conv2DTranspose, Lambda, Layer
from keras import backend as K
from evaluation import generate_results, predict_model
import numpy as np
from keras.models import Model, load_model
from soft_attention import generate_embeddings_attention, generate_embeddings_hard_attetion
from constants import POLITICAL_PARTIES, COUNTRY_PARTIES
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
import tensorflow as tf
from custom_classes import L2Norm


def replace_sigmoid_with_softmax(config, classes, folder, loaded_model):
    loaded_model.summary()
    last_layer = loaded_model.get_layer('dr_spacnn_0')
    input_weights = []
    input_weights.append(loaded_model.input)
    dense_sub = Dense(len(classes), name="main_dense")(last_layer.output)
    dense_sub = Activation('sigmoid', name='main')(dense_sub)
    output_weights = [dense_sub]
    model = Model(inputs=input_weights, outputs=output_weights)
    plot_model(model, to_file=folder + '/model.png')
    return model
def multi_label_architecture(config, softmax_len, folder):
    return create_model_late_fusion(config, softmax_len, folder)

def adrian_architecture(config, softmax_len, softmax_len_sub, folder, multilingua=None):
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    emb_layer = generate_embedding_matrix(config, folder,multilingua)
    emb_layer.name = "no_freezed_emb"
    embed_1 = emb_layer(x_1)
    joined = generate_cnns(config, embed_1, "no_freeze")
    drop1 = Dropout(config['dropout'])(joined)
    softmax_1 = generate_second_part_after_cnns(config, drop1, softmax_len)
    print str(joined.shape)
    print str(type(embed_1))
    #joined = Reshape((1, 300))(joined)
    #print str(joined.shape)
    #joined_repeated = K.repeat_elements(joined, -1, config['max_phrase_length'])
    #joined_repeated = K.repeat(joined, 60)
    if config['use_normalizer']:
        joined = L2Norm()(joined)
    joined_repeated = Lambda(lambda x: K.repeat(x, 60) )(joined)
    print str(joined_repeated.shape)
    print str(type(joined_repeated))
    reshape_1 = Reshape((config['max_phrase_length'],config['embedding_size_1'],1))(embed_1)
    reshape_2 = Reshape((config['max_phrase_length'],config['embedding_size_1'], 1))(joined_repeated)
    merged_vector = concatenate([reshape_1, reshape_2], axis=-1)
    convs = []
    for i, fsz in enumerate(config['filter_sizes']):
        conv = Conv2D(config['num_filters'], fsz, config['embedding_size_1'],input_shape=(config['max_phrase_length'],config['embedding_size_1'],2))(merged_vector)
        if  config['batch_normalization']:
            conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        gmp = GlobalMaxPooling2D()(conv)
        convs.append(gmp)
    joined = concatenate(convs)
    drop1 = Dropout(config['dropout'])(joined)
    softmax_sub = generate_second_part_after_cnns(config, drop1, softmax_len_sub, "_subdomain")
    input_weights = [x_1]
    output_weights = [softmax_1, softmax_sub]
    model = Model(inputs=input_weights, outputs=output_weights)
    model.summary()
    return model

def multi_stream_single_loss(config, softmax_len, folder,  multilingua=None):
    loaded_model = load_model(config['freezed_model'] + "best_model.hdf5")
    for layer in loaded_model.layers[:5]:
        layer.trainable = False
    loaded_model.summary()
    last_layer = loaded_model.get_layer('dr_spacnn_0')
    freezed_cnn_features = loaded_model.get_layer('concatenate_f1')
    input_weights = []
    input_weights.append(loaded_model.input)
    embed_1 = None
    if config['share_embedding']:
        em = loaded_model.get_layer('embedding_1')
        em.trainable = True
        embed_1 = em.output
    else:
        emb_layer = generate_embedding_matrix(config, folder,multilingua)
        emb_layer.name = "no_freezed_emb"
        embed_1 = emb_layer(loaded_model.input)
    joined = generate_cnns(config, embed_1, "no_freeze")
    if config['concatenate_cnn_features']:
        if config['use_normalizer']:
            normalize_1 = L2Norm()(joined)
            normalize_2 = L2Norm()(freezed_cnn_features.output)
            joined = concatenate([normalize_1, normalize_2])
        else:
            joined = concatenate([joined,freezed_cnn_features.output], axis=1, name="concatenate_f2_not_freezed")
    drop1 = Dropout(config['dropout'], name="dropout_f1_not_freezed")(joined)
    dense_relu = Dense(512, name="first_dense_not_freezed")(drop1)
    if config["batch_normalization_relu_soft"]:
        dense_relu = BatchNormalization(name="batch_normalization_f1" + "_not_freezed")(dense_relu)
    dense_relu = Activation(config["denses"][0], name="activation_f1_not_freezed")(dense_relu)
    dense_relu  = Dropout(config['dropout'], name="dropout_f2_not_Freezed")(dense_relu)
    concatenated_lul = concatenate([dense_relu, last_layer.output], name="concatenate_f3_not_freezed", axis=1)
    if config['dense_after_feature_concatenation']:
        concatenated_lul = Dense(512, name="densen_f6_not_freezed")(concatenated_lul)
        concatenated_lul = Activation("relu", name="activation_f6_not_freezed")(concatenated_lul)
        concatenated_lul  = Dropout(config['dropout'], name="dropout_f6_not_Freezed")(concatenated_lul)
    dense_sub = Dense(softmax_len, activation=config['last_activation'], name="main")(concatenated_lul)
    output_weights = [dense_sub]
    model = Model(inputs=input_weights, outputs=output_weights)
    plot_model(model, to_file=folder + '/model.png')
    return model
def custom_loss_experiment_domain_freezed(config, softmax_len_c1, softmax_len_c2, folder, multilingua=None):
    loaded_model = load_model(config['freezed_model'] + "best_model.hdf5")
    for layer in loaded_model.layers[:5]:
        layer.trainable = False
    loaded_model.summary()
    last_layer = loaded_model.get_layer('dropout_2')
    freezed_cnn_features = loaded_model.get_layer('concatenate_1')
    input_weights = []
    #input_layer = model.get_layer('phrases')
    #print input_layer
    input_weights.append(loaded_model.input)
    if config['class'] == "subdomain":
        label_layer_1 = Input(shape=(softmax_len_c2,), name='gt_domain')
        label_layer_2 = Input(shape=(softmax_len_c1,), name='gt_subdomain')
    else:
        label_layer_1 = Input(shape=(softmax_len_c1,), name='gt_domain')
        label_layer_2 = Input(shape=(softmax_len_c2,), name='gt_subdomain')
    input_weights.append(label_layer_1)
    input_weights.append(label_layer_2)
    emb_layer = generate_embedding_matrix(config, folder,multilingua)
    emb_layer.name = "no_freezed_emb"
    embed_1 = emb_layer(loaded_model.input)
    #joined = []
    #joined.append(generate_cnns(config, embed_1))
    joined = generate_cnns(config, embed_1, "no_freeze")
    if config['concatenate_cnn_features']:
        joined = concatenate([joined,freezed_cnn_features.output], axis=1, name="concatenate_f2_not_freezed")
    drop1 = Dropout(config['dropout'], name="dropout_f1_not_freezed")(joined)
    dense_relu = Dense(512, name="first_dense_not_freezed")(drop1)
    if config["batch_normalization_relu_soft"]:
        dense_relu = BatchNormalization(name="batch_normalization_f1" + "_not_freezed")(dense_relu)
    dense_relu = Activation(config["denses"][0], name="activation_f1_not_freezed")(dense_relu)
    dense_relu  = Dropout(config['dropout'], name="dropout_f2_not_Freezed")(dense_relu)
    concatenated_lul = concatenate([dense_relu, last_layer.output], name="concatenate_f3_not_freezed", axis=1)
    if config['class'] == "subdomain":
        dense_sub = Dense(softmax_len_c1, activation=config['last_activation'], name="secondary")(concatenated_lul)
    else:
        dense_sub = Dense(softmax_len_c2, activation=config['last_activation'], name="secondary")(concatenated_lul)
    output_weights = [loaded_model.output, dense_sub]
    model = Model(inputs=input_weights, outputs=output_weights)
    plot_model(model, to_file=folder + '/model.png')
    loss = K.mean((categorical_crossentropy(loaded_model.output, label_layer_1)*config['weight_1']) + (categorical_crossentropy(dense_sub, label_layer_2)*config['weight_2']))
    model.add_loss(loss)
    return model
def custom_loss_experiment(config, softmax_len, softmax_len_subdomain, folder, multilingua=None):
    input_weights = []
    emb_layer = generate_embedding_matrix(config, folder,multilingua)
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    input_weights.append(x_1)
    embed_1 = emb_layer(x_1)
    joined = []
    joined.append(generate_cnns(config, embed_1))
    if config['previous_phrase']:
        x_2 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_phrases')
        input_weights.append(x_2)
        embed_2 = emb_layer(x_2)
        joined.append(generate_cnns(config, embed_2))
    if config['previous_previous']:
        x_3 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_previous_phrases')
        input_weights.append(x_3)
        embed_3 = emb_layer(x_3)
        joined.append(generate_cnns(config, embed_3))
    if config['post_phrase']:
        x_4 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='post_phrases')
        input_weights.append(x_4)
        embed_4 = emb_layer(x_4)
        joined.append(generate_cnns(config, embed_4, "post_phrases"))
    if len(joined) > 1:
        joined = concatenate(joined, axis=1)
    else:
        joined = joined[0]

    if config['party_as_one_hot'] and config['party'] and not config['party_as_rile_score'] and not config['party_as_std_mean']:
        if not config['multilingual']:
            x_3 = Input(shape=(len(POLITICAL_PARTIES),), dtype='float32', name='party')
            input_weights.append(x_3)
        else:
            x_3 = Input(shape=(len(COUNTRY_PARTIES[config['language']]),), dtype='float32', name='party')
            input_weights.append(x_3)
    elif config['party'] and not config['party_as_rile_score'] and not config['party_as_one_hot'] and not config['party_as_std_mean'] and not config['party_as_deconv']:
        x_3 = Input(shape=(model_size,), dtype='float32', name='party')
        input_weights.append(x_3)
    elif config['party'] and not config['party_as_one_hot'] and (config['party_as_rile_score'] or config[
        'party_as_std_mean']):
        x_3 = Input(shape=(1,), dtype='float32', name='party')
        input_weights.append(x_3)
    elif config['party'] and config['party_as_deconv']:
        x_3 = Input(shape=(1,10,10), dtype='float32', name='party')
        input_weights.append(x_3)        
    if config['party'] and config['dropout_before_party']:
        drop1 = Dropout(config['dropout'])(joined)
        if not config['party_as_deconv']:
            drop1 = concatenate([drop1, x_3], axis=1)
        else:
            joined = Conv2DTranspose(200,(2,2))(x_3)
            joined = Flatten()(joined)
    elif config['party']:
        if not config['party_as_deconv']:
            joined = concatenate([joined, x_3], axis=1)
        else:
            joined = Conv2DTranspose(200,(2,2))(x_3)
            joined = Flatten()(joined)
        drop1 = Dropout(config['dropout'])(joined)
    else:
        drop1 = Dropout(config['dropout'], name="dropout1")(joined)
        #pass
    if config["denses"][0] == "linear":
        dense_relu = Dense(512)(drop1)
    else:
        dense_relu = Dense(512, name="first_dense")(drop1)
        if config["batch_normalization_relu_soft"]:
            dense_relu = BatchNormalization()(dense_relu)
        dense_relu = Activation(config["denses"][0])(dense_relu)
        dense_relu  = Dropout(config['dropout'], name="dropout2")(dense_relu)
    for dense in config["denses"][1:]:
        dense_relu = Dense(512, activation=dense)(dense_relu)
        dense_relu = Dropout(config['dropout'])(dense_relu)
    print "Is binary?" + str(config['binary'])
    print "Softmax len " + str(softmax_len)
    if config['binary']:
        softmax_len = 2
    print "Last activation is " + config['last_activation']
    dense_domain = Dense(softmax_len, activation=config['last_activation'], name="main")(dense_relu)
    dense_sub = Dense(softmax_len_subdomain, activation=config['last_activation'], name="secondary")(dense_relu)
    output_weights = [dense_domain, dense_sub]
    label_layer_1 = Input(shape=(softmax_len,), name='gt_domain')
    label_layer_2 = Input(shape=(softmax_len_subdomain,), name='gt_subdomain')
    #dense_fake_1 = Dense(1) (label_layer_1)
    #dense_fake_2 = Dense(1)(label_layer_2)
    input_weights.append(label_layer_1)
    input_weights.append(label_layer_2)
    #output_weights.append(dense_fake_1)
    #output_weights.append(dense_fake_2)
    model = Model(inputs=input_weights, outputs=output_weights)
    loss = K.mean((categorical_crossentropy(dense_domain, label_layer_1)*config['weight_1']) + (categorical_crossentropy(dense_sub, label_layer_2)*config['weight_2']))
    model.add_loss(loss)
    return model


def create_model_channel(config, softmax_len, folder, multilingua=None):
    input_weights = []
    emb_layer = generate_embedding_matrix(config, folder,multilingua)
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    input_weights.append(x_1)
    embed_1 = emb_layer(x_1)
    x_2 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_phrases')
    input_weights.append(x_2)
    embed_2 = emb_layer(x_2)
    joined = generate_prev_channel_cnns(config, embed_1, embed_2)
    if config['party']:
        x_3 = Input(shape=(len(COUNTRY_PARTIES[config['language']]),), dtype='float32', name='party')
        input_weights.append(x_3)
    drop1 = Dropout(config['dropout'])(joined)
    if config['party']:
        drop1 = concatenate([drop1, x_3], axis=1)
    dense_1 = generate_second_part_after_cnns(config, drop1, softmax_len)
    output_weights = [dense_1]
    model = Model(inputs=input_weights, outputs=output_weights)
    return model


def generate_cnns(config, embed, layer_names=""):
    convs = []
    for i, fsz in enumerate(config['filter_sizes']):
        conv = Conv1D(config['num_filters'], fsz, name="con2d_method_"+str(i) + layer_names , input_shape=(config['max_phrase_length'],))(embed)
        if config['batch_normalization']:
            conv = BatchNormalization(name="batch_normalization_1"+str(i)  + layer_names)(conv)
        conv = Activation("relu", name="activation_1"+str(i)  + layer_names)(conv)
        gmp = GlobalMaxPooling1D(name="pooling_" + str(i)+ layer_names)(conv)
        convs.append(gmp)
    joined_1 = concatenate(convs, axis=1,name="concatenate_f1"  + layer_names)
    return joined_1


def generate_prev_channel_cnns(config, incoming_1, incoming_2):
    reshape_1 = Reshape((config['max_phrase_length'],config['embedding_size_1'],1))(incoming_1)
    reshape_2 = Reshape((config['max_phrase_length'],config['embedding_size_1'], 1))(incoming_2)
    merged_vector = concatenate([reshape_1, reshape_2], axis=-1)
    convs = []
    for i, fsz in enumerate(config['filter_sizes']):
        conv = Conv2D(config['num_filters'], fsz, config['embedding_size_1'],input_shape=(config['max_phrase_length'],config['embedding_size_1'],2))(merged_vector)
        if  config['batch_normalization']:
            conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        gmp = GlobalMaxPooling2D()(conv)
        convs.append(gmp)
    joined = concatenate(convs)
    return joined



def generate_second_part_after_cnns(config, incoming_layer, softmax_len, layer_names=""):
    layer_name = "dense_spacnn_0" + layer_names
    if config["denses"][0] == "linear":
        dense_relu = Dense(512, name=layer_name)(incoming_layer)
    else:
        dense_relu = Dense(512, name=layer_name)(incoming_layer)
        if config["batch_normalization_relu_soft"]:
            dense_relu = BatchNormalization(name="bn_spacnn_0"+ layer_names)(dense_relu)
        dense_relu = Activation(config["denses"][0])(dense_relu)
        dense_relu  = Dropout(config['dropout'], name="dr_spacnn_0"+ layer_names)(dense_relu)
    for dense in config["denses"][1:]:
        dense_relu = Dense(512,name="dense_spacnn_1", activation=dense)(dense_relu)
        dense_relu = Dropout(config['dropout'], name="dr_spacnn_1"+ layer_names)(dense_relu)
    print "Is binary?" + str(config['binary'])
    print "Softmax len " + str(softmax_len)
    if config['binary']:
        softmax_len = 2
    print "Last activation is " + config['last_activation']
    dense = Dense(softmax_len, activation=config['last_activation'], name="main"+ layer_names)(dense_relu)
    return dense




def create_model_late_fusion(config, softmax_len, folder, multilingua=None):
    input_weights = []
    emb_layer = generate_embedding_matrix(config, folder,multilingua)
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    input_weights.append(x_1)
    embed_1 = emb_layer(x_1)
    if config['attention']:
        embed_1 = generate_embeddings_attention(config, embed_1)
    elif config['hard_attention']:
        embed_1 = generate_embeddings_hard_attetion(config, embed_1)
    joined = []
    joined.append(generate_cnns(config, embed_1))
    if config['previous_phrase']:
    	x_2 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_phrases')
    	input_weights.append(x_2)
    	embed_2 = emb_layer(x_2)
    	joined.append(generate_cnns(config, embed_2, "previous_phrases"))
    if config['previous_previous']:
     	x_3 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_previous_phrases')
     	input_weights.append(x_3)
    	embed_3 = emb_layer(x_3)
    	joined.append(generate_cnns(config, embed_3))
    if config['post_phrase']:
        x_4 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='post_phrases')
        input_weights.append(x_4)
        embed_4 = emb_layer(x_4)
        joined.append(generate_cnns(config, embed_4, "post_phrases"))
    if len(joined) > 1:
    	joined = concatenate(joined, axis=1)
    else:
        joined = joined[0]

    if config['party_as_one_hot'] and config['party'] and not config['party_as_rile_score'] and not config['party_as_std_mean']:
        if not config['multilingual']  and config['language'] != 'english':
            x_3 = Input(shape=(len(POLITICAL_PARTIES),), dtype='float32', name='party')
            input_weights.append(x_3)
        else:
            x_3 = Input(shape=(len(COUNTRY_PARTIES[config['language']]),), dtype='float32', name='party')
            input_weights.append(x_3)
    elif config['party'] and not config['party_as_rile_score'] and not config['party_as_one_hot'] and not config['party_as_std_mean'] and not config['party_as_deconv']:
        x_3 = Input(shape=(model_size,), dtype='float32', name='party')
        input_weights.append(x_3)
    elif config['party'] and not config['party_as_one_hot'] and (config['party_as_rile_score'] or config[
        'party_as_std_mean']):
        x_3 = Input(shape=(1,), dtype='float32', name='party')
        input_weights.append(x_3)
    elif config['party'] and config['party_as_deconv']:
        x_3 = Input(shape=(1,10,10), dtype='float32', name='party')
        input_weights.append(x_3)        
    if config['party'] and config['dropout_before_party']:
        drop1 = Dropout(config['dropout'])(joined)
        if not config['party_as_deconv']:
            drop1 = concatenate([drop1, x_3], axis=1)
        else:
            joined = Conv2DTranspose(200,(2,2))(x_3)
            joined = Flatten()(joined)
    elif config['party']:
        if not config['party_as_deconv']:
            joined = concatenate([joined, x_3], axis=1)
        else:
            joined = Conv2DTranspose(200,(2,2))(x_3)
            joined = Flatten()(joined)
        drop1 = Dropout(config['dropout'])(joined)
    else:
        drop1 = Dropout(config['dropout'])(joined)
        #pass
    dense_1 = generate_second_part_after_cnns(config, drop1, softmax_len)
    output_weights = [dense_1]
    model = Model(inputs=input_weights, outputs=output_weights)
    return model


def generate_previous_phrase_as_attention(config, softmax_len, folder, multilingua=None):
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    x_2 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='previous_phrases')
    emb_layer = generate_embedding_matrix(config, folder)
    input_weights = []
    input_weights.append(x_1)
    input_weights.append(x_2)
    embed_1 = emb_layer(x_1)
    embed_2 = emb_layer(x_2)
    embed_concatenation = Concatenate(axis=1)([embed_1, embed_2])
    gru = Bidirectional(GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                        recurrent_dropout=config['recurrent_dropout']))(embed_concatenation)
    dense_att_1 = TimeDistributed(Dense(config['gru_units'], activation=config['attention_activation']))(gru)
    dense_att_2 = TimeDistributed(Dense(1))(dense_att_1)
    reshape_distributed = Reshape((config['max_phrase_length']*2,))(dense_att_2)
    attention = Activation('softmax')(reshape_distributed)
    reshape_att = Reshape((config['max_phrase_length']*2, 1), name='reshape_att')(attention)
    apply_att = Multiply()([embed_1, reshape_att])
    joined = []
    new_embed = generate_cnns(config, apply_att)
    drop1 = Dropout(config['dropout'])(new_embed)
    dense_1 = generate_second_part_after_cnns(config, drop1, softmax_len)
    output_weights = [dense_1]
    model = Model(inputs=input_weights, outputs=output_weights)
    return model

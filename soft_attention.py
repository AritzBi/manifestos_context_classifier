from keras.layers import Dense, GRU, TimeDistributed, Multiply, Bidirectional, Average, Maximum, Reshape, Input, \
    Activation, BatchNormalization, Dropout, Dot, ThresholdedReLU
from keras.models import Model, load_model
from data_preprocessing import generate_embedding_matrix

def generate_embeddings_attention(config, incoming_layer):
    if config['bidirectional_attention']:
        gru = Bidirectional(GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                            recurrent_dropout=config['recurrent_dropout']))(incoming_layer)
    else:
        gru = GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                            recurrent_dropout=config['recurrent_dropout'])(incoming_layer)
    dense_att_1 = TimeDistributed(Dense(config['gru_units'], activation=config['attention_activation']))(gru)
    dense_att_2 = TimeDistributed(Dense(1))(dense_att_1)
    # to undo the time distribution and have 1 value for each action
    reshape_distributed = Reshape((config['max_phrase_length'],))(dense_att_2)
    attention = Activation('softmax')(reshape_distributed)
    #so we can multiply it with embeddings
    reshape_att = Reshape((config['max_phrase_length'], 1), name='reshape_att')(attention)
    if config['merge_mode'] == 'multiply':
        apply_att = Multiply()([incoming_layer, reshape_att])
    elif config['merge_mode'] == 'average':
        apply_att = Average()([incoming_layer, reshape_att])
    elif config['merge_mode'] == 'maximum':
        apply_att = Maximum()([incoming_layer, reshape_att])
    return apply_att
    #apply the attention to the embeddings
    #add channel dimension for the CNNs
    #reshape = Reshape((config['max_phrase_length'], config['embedding_size_1']), name='reshape')(apply_att)
    #return reshape


def generate_embeddings_hard_attetion(config, incoming_layer):
    if config['bidirectional_attention']:
        gru = Bidirectional(GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                                recurrent_dropout=config['recurrent_dropout']))(incoming_layer)
    else:
        gru = GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                  recurrent_dropout=config['recurrent_dropout'])(incoming_layer)

    dense_att_1 = TimeDistributed(Dense(config['gru_units'], name='dense_att_1'))(gru)
    att_1_act = ThresholdedReLU(theta=0.8)(dense_att_1)
    # total units = 1 * INPUT_ACTIONS
    dense_att_2 = TimeDistributed(Dense(1))(att_1_act)
    # to undo the time distribution and have 1 value for each action
    reshape_distributed = Reshape((config['max_phrase_length'],))(dense_att_2)
    attention = Activation('softmax')(reshape_distributed)
    # so we can multiply it with embeddings
    reshape_att = Reshape((config['max_phrase_length'], 1), name='reshape_att')(attention)
    # apply the attention to the embeddings
    apply_att = Multiply()([incoming_layer, reshape_att])
    return apply_att

def gru_attention_model(config, softmax_len, folder):
    input_weights = []
    emb_layer = generate_embedding_matrix(config, folder)
    x_1 = Input(shape=(config['max_phrase_length'],), dtype='int32', name='phrases')
    input_weights.append(x_1)
    embed_1 = emb_layer(x_1)
    gru = Bidirectional(GRU(config['gru_units'], return_sequences=True, dropout=config['gru_dropout'],
                            recurrent_dropout=config['recurrent_dropout']))(embed_1)
    dense_att_1 = TimeDistributed(Dense(config['gru_units'], activation=config['attention_activation']))(gru)
    dense_att_2 = TimeDistributed(Dense(1))(dense_att_1)
    reshape_distributed = Reshape((config['max_phrase_length'],))(dense_att_2)
    attention = Activation('softmax')(reshape_distributed)
    apply_attention = Dot(axes=(1, 1))([attention, gru])
    dense_relu = Dense(512)(apply_attention)
    if config["batch_normalization_relu_soft"]:
        dense_relu = BatchNormalization()(dense_relu)
    dense_relu = Activation(config["denses"][0])(dense_relu)
    dense_relu  = Dropout(config['dropout'])(dense_relu)
    print "Is binary?" + str(config['binary'])
    print "Softmax len " + str(softmax_len)
    if config['binary']:
        softmax_len = 2
    print "Last activation is " + config['last_activation']
    dense = Dense(softmax_len, activation=config['last_activation'], name="main")(dense_relu)
    output_weights = [dense]
    model = Model(inputs=input_weights, outputs=output_weights)
    return model
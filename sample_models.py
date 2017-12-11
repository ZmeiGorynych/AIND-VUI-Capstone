from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)


# A Keras decoder with attention from
# https://raw.githubusercontent.com/datalogue/keras-attention/master/models/custom_recurrents.py
from custom_recurrents import AttentionDecoder

NUM_TIME_SLICES = 1000

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    tmp = input_data
    for i in range(recur_layers):
        rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn' + str(i))(tmp)
        # TODO: Add batch normalization
        tmp = BatchNormalization()(rnn)


    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(tmp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bi_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2))(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(bi_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense =  TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_deep_bidir_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, bidir_layers, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    tmp = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    for _ in range(bidir_layers):
        bi_rnn = Bidirectional(GRU(units, activation='relu',
            return_sequences=True, implementation=2))(tmp)
        tmp = BatchNormalization()(bi_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(tmp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def simple_decoder_model(input_dim, units, output_dim=29):

    input_data = Input(name='the_input', shape=(NUM_TIME_SLICES, input_dim))
    rnn = GRU(units, return_sequences=True, implementation=2)(input_data)
    decoded = AttentionDecoder(units,output_dim)(rnn)
    y_pred = Activation('softmax', name='softmax')(decoded)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x:x
    print(model.summary())
    return model

def cnn_deep_bidir_rnn_attention_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, bidir_layers, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(NUM_TIME_SLICES, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    tmp = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    for _ in range(bidir_layers):
        bi_rnn = Bidirectional(
                    GRU(units, activation='relu',
                        return_sequences=True, implementation=2),
                    merge_mode='concat')(tmp)
        tmp = BatchNormalization()(bi_rnn)

    # Bidirectional(LSTM(encoder_units, return_sequences=True),
    #               name='bidirectional_1',
    #               merge_mode='concat',
    #               trainable=trainable
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    decoded = AttentionDecoder(units,output_dim)(tmp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(decoded)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_deep_bidir_rnn_attention_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, bidir_layers, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(NUM_TIME_SLICES, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    tmp = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    for _ in range(bidir_layers):
        bi_rnn = Bidirectional(
                    GRU(units, activation='relu',
                        return_sequences=True, implementation=2),
                    merge_mode='concat')(tmp)
        tmp = BatchNormalization()(bi_rnn)

    # Bidirectional(LSTM(encoder_units, return_sequences=True),
    #               name='bidirectional_1',
    #               merge_mode='concat',
    #               trainable=trainable
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    decoded = AttentionDecoder(units,output_dim)(tmp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(decoded)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_deep_bidir_rnn_dropout_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, bidir_layers, output_dim=29,drop_rate=0):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    tmp = Dropout(drop_rate)(conv_1d)
    # Add batch normalization
    tmp = BatchNormalization(name='bn_conv_1d')(tmp)
    # Add a recurrent layer
    for _ in range(bidir_layers):
        bi_rnn = Bidirectional(GRU(units, activation='relu',
            return_sequences=True, implementation=2))(tmp)
        drop = Dropout(drop_rate)(bi_rnn)
        tmp = BatchNormalization()(drop)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(tmp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim):
    """ Build a deep network for speech
    """
    # note that cnn_deep_bidir_rnn_dropout_model is used here for the first time,
    # as per the rubric
    return cnn_deep_bidir_rnn_dropout_model(input_dim,
                             filters=200,
                             kernel_size=11,
                             conv_stride=2,
                             conv_border_mode='valid',
                             units=200,
                             bidir_layers=3,
                             output_dim=29,
                            drop_rate=0.1)

if __name__=='__main__':
    spectrogram = True
    #model_5 = simple_model(input_dim=161,units=200,output_dim=29)
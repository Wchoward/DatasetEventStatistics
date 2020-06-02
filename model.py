import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Activation, Flatten, Conv2D, Dense, Dropout, LSTM


class MainModel:

    def __init__(self):

        pass

    def model_build(self, module_1, module_2, input_1, input_2):

        output_1 = 0
        output_2 = 0
        if module_1 == 'cnn':
            input_1 = tf.expand_dims(input_1, -1)
            output_1 = self.cnn_module(input_1)
        elif module_1 == "lstm":
            output_1 = self.rnn_module(input_1)
        elif module_1 == 'attention':
            output_1 = self.attention_module(input_1)

        if module_2 == 'cnn':
            # input_2 = tf.expand_dims(input_2, -1)
            # input_2 = tf.keras.layers.Lambda(expand_dim)(input_2)
            input_2 = tf.expand_dims(input_2, -1)
            output_2 = self.cnn_module(input_2)
        elif module_2 == 'lstm':
            output_2 = self.rnn_module(input_2)
        elif module_2 == 'attention':
            output_2 = self.attention_module(input_2)

        output = self.output_module(output_1, output_2)

        return output

    def cnn_module(self, input):

        with tf.name_scope("information_confusion_module"):
            conv_1 = Conv2D(8, (5, 5), strides=(2, 2), padding="same", data_format="channels_last",
                            input_shape=(tf.shape(input)[1], tf.shape(input)[2], 1))(input)
            conv_1 = BatchNormalization()(conv_1)
            conv_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv_1)
            conv_1 = Activation("relu")(conv_1)

            conv_2 = Conv2D(8, (5, 5), strides=(2, 2), padding="same", data_format="channels_last")(conv_1)
            conv_2 = BatchNormalization()(conv_2)
            conv_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv_2)
            conv_2 = Activation("relu")(conv_2)

            conv_3 = Conv2D(16, (10, 10), strides=(2, 2), padding="valid", data_format="channels_last")(conv_2)
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv_3)
            conv_3 = Activation("relu")(conv_3)

            output = Flatten()(conv_3)
            output = Dense(64, activation='relu')(output)

        return output

    def rnn_module(self, input):

        with tf.name_scope("rnn_information_confusion_module"):
            lstm_1 = tf.keras.layers.Bidirectional(LSTM(32, return_sequences=True))(input)
            output = tf.keras.layers.Bidirectional(LSTM(32))(lstm_1)

            return output

    def attention_module(self, input):

        # tf.Variable()

        pass

    def concatenate(self, input):
        return tf.concat(input, axis=-1)

    def output_module(self, original_text, event):

        with tf.name_scope("output_module"):
            # concatenation_vector = tf.concat([original_text, event], axis=-1)
            concatenation_vector = tf.keras.layers.Lambda(self.concatenate)([original_text, event])
            # FC1
            fc_1 = Dense(64, activation='relu')(concatenation_vector)
            fc_1 = Dropout(0.5)(fc_1)

            # FC2
            fc_2 = Dense(2, activation='softmax')(fc_1)

            return fc_2

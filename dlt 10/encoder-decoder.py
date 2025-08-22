import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
input_texts = [
    "hello", "how are you", "good morning", "thank you", "i love you"
]
target_texts = [
    "hola", "como estas", "buenos dias", "gracias", "te amo"
]

target_texts = ["<start> " + text + " <end>" for text in target_texts]
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_word_index = input_tokenizer.word_index
num_encoder_tokens = len(input_word_index) + 1

target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_word_index = target_tokenizer.word_index
num_decoder_tokens = len(target_word_index) + 1
max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding="post")
decoder_input_data = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_decoder_seq_length, padding="post")
decoder_target_data = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_decoder_seq_length, padding="post")
decoder_target_data = tf.keras.utils.to_categorical(decoder_target_data, num_decoder_tokens)
encoder_inputs = Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(num_encoder_tokens, 64)(encoder_inputs)
encoder_lstm = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None,))
dec_emb = tf.keras.layers.Embedding(num_decoder_tokens, 64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=16, epochs=500, verbose=0)

print("✅ Training finished!")
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(64,))
decoder_state_input_c = Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = tf.keras.layers.Embedding(num_decoder_tokens, 64)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)
reverse_target_index = {i: word for word, i in target_word_index.items()}
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_index.get(sampled_token_index, '')

        if (sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()
test_sentence = "hello"
test_seq = pad_sequences(input_tokenizer.texts_to_sequences([test_sentence]), maxlen=max_encoder_seq_length, padding="post")
translation = decode_sequence(test_seq)
print(f"English: {test_sentence}")
print(f"Spanish Translation: {translation}")

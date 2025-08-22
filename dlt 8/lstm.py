import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
conversations = {
    "hi": "hello",
    "how are you": "i am fine",
    "what is your name": "i am a chatbot",
    "bye": "goodbye",
    "thank you": "you are welcome"
}

questions = list(conversations.keys())
answers = list(conversations.values())
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1
X_seq = tokenizer.texts_to_sequences(questions)
Y_seq = tokenizer.texts_to_sequences(answers)

max_len = max(max(len(seq) for seq in X_seq), max(len(seq) for seq in Y_seq))
X = pad_sequences(X_seq, maxlen=max_len, padding='post')
Y = pad_sequences(Y_seq, maxlen=max_len, padding='post')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Y must be single token outputs -> use only first token of answer
Y_first_word = np.array([seq[0] if len(seq) > 0 else 0 for seq in Y])
model.fit(X, Y_first_word, epochs=300, verbose=0)
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def generate_reply(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)
    word_id = np.argmax(pred)
    return reverse_word_index.get(word_id, "")
print("User: hi")
print("Bot:", generate_reply("hi"))

print("User: how are you")
print("Bot:", generate_reply("how are you"))

print("User: what is your name")
print("Bot:", generate_reply("what is your name"))

print("User: bye")
print("Bot:", generate_reply("bye"))

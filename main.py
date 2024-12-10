import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.utils import plot_model
import pickle
from temizle import temizle
from sklearn.metrics import classification_report

# NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Veri setini yükle
data = pd.read_csv('yeni_veriler.csv')
data = data.dropna()

# Yorumları temizle
data['cleaned_comment'] = data['comment'].apply(temizle)

# Etiketleri sayısal hale getir
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Yorumları tokenleştir ve Word2Vec modelini eğit
data['cleaned_comment'] = data['cleaned_comment'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
tokenized_comments = [word_tokenize(comment) for comment in data['cleaned_comment']]
word2vec_model = Word2Vec(sentences=tokenized_comments, vector_size=100, window=5, min_count=1, workers=4)

# Yorumları sayısal hale getir ve padding yap
max_len = 100
X = []
for comment in tokenized_comments:
    # Word2Vec vektörlerini al
    comment_vectors = []
    for word in comment[:max_len]:
        if word in word2vec_model.wv:
            comment_vectors.append(word2vec_model.wv[word])
        else:
            comment_vectors.append(np.zeros(100))

    # Padding
    while len(comment_vectors) < max_len:
        comment_vectors.append(np.zeros(100))

    # Vektörü numpy array'e çevir
    comment_vectors = np.array(comment_vectors)
    X.append(comment_vectors)

# Numpy arraye dönüştür
X = np.array(X)

# Temel model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(max_len, 100)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Optimum model
optimized_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(max_len, 100))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optimum Modeli derle
optimized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin görselini kaydet
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Verileri eğit ve test setlerine ayır
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Optimum modeli eğit
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
optimized_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Modeli kaydet
model.save('yorum_analiz_model.keras')
word2vec_model.save('yorum_analiz_word2vec.model')
with open('yorum_analiz_labelencoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Optimum modeli kaydet
optimized_model.save('yorum_analiz_model_o.keras')
word2vec_model.save('yorum_analiz_word2vec_o.model')
with open('yorum_analiz_labelencoder_o.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Modellerin performans tablosu
y_pred_model = model.predict(X_test).argmax(axis=1)
y_pred_model_optimized = optimized_model.predict(X_test).argmax(axis=1)
print("Temel modelin performans tablosu")
print(classification_report(y_test, y_pred_model, target_names=label_encoder.classes_))
print("Optimum modelin performans tablosu")
print(classification_report(y_test, y_pred_model_optimized, target_names=label_encoder.classes_))
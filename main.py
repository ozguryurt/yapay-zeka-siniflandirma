import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
from tkinter import simpledialog, messagebox
import nltk
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import re

# nltk veri setlerini yükleme
nltk.download('stopwords')

# CSV dosyasını yükle
df = pd.read_csv('magazayorum_cleanedd.csv')  # Dosya yolunu gerektiği gibi güncelle

# Türkçe stopwords ve stemmer
stop_words = set(stopwords.words('turkish'))
stemmer = TurkishStemmer()

# Yorumları temizleme ve işleme fonksiyonu
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Özel karakterleri kaldırma
    tokens = text.lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Tüm yorumları ön işlemden geçir
df['comment'] = df['comment'].apply(preprocess_text)

# Yorumlar ve etiketler
X = df['comment']
y = df['label']

# Etiketleri sayısal verilere dönüştürme
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizasyon
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Yapay sinir ağı modeli
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 sınıf olduğu için '3' kullanılıyor

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Tkinter ile arayüz
def classify_comment():
    while True:
        input_text = simpledialog.askstring("Yorum Giriniz", "Lütfen yorumunuzu giriniz:")
        if not input_text:
            break  # Kullanıcı 'Cancel' derse veya boş bırakırsa döngüden çıkar

        processed_text = preprocess_text(input_text)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_text)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        messagebox.showinfo("Sonuç", f"Girilen yorum: {predicted_label}")


# Arayüzü başlatma
root = tk.Tk()
root.withdraw()  # Ana pencereyi gizle, sadece giriş penceresini göster

classify_comment()
root.mainloop()

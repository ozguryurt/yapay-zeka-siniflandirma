import threading
import tkinter as tk
from tkinter import ttk
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from video import get_video_data
from temizle import temizle
from gensim.models import Word2Vec
import numpy as np
import re
import os

# Model dosyaları
model = load_model('yorum_analiz_model.keras')
word2vec_model = Word2Vec.load('yorum_analiz_word2vec.model')
with open('yorum_analiz_labelencoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def get_video_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    else:
        return None

def analiz_et(yorum):
    max_len = 100
    vector_size = 100
    try:
        temizlenmis_yorum = temizle(yorum)
        comment_vectors = []
        for word in temizlenmis_yorum[:max_len]:
            if word in word2vec_model.wv:
                comment_vectors.append(word2vec_model.wv[word])
            else:
                comment_vectors.append(np.zeros(vector_size))
        while len(comment_vectors) < max_len:
            comment_vectors.append(np.zeros(vector_size))
        vektorel_yorum = np.array(comment_vectors)
        vektorel_yorum = np.expand_dims(vektorel_yorum, axis=0)
        tahmin = model.predict(vektorel_yorum, verbose=0)
        tahmin_sinifi = np.argmax(tahmin, axis=1)
        etiket = label_encoder.inverse_transform(tahmin_sinifi)
        return etiket[0]
    except Exception as e:
        return None
def yorumlari_analiz_et_ve_tabloyu_guncelle(datatable, comments_data, status_label, csv_button, link, stop_event):
    status_label.config(text="Yorumlar getirildi, analiz ediliyor...")
    for comment in comments_data:
        if stop_event.is_set():
            status_label.config(text="İşlem iptal edildi.")
            return
        label = analiz_et(comment['content'])
        datatable.insert("", "end", values=(get_video_id(link), comment['username'], comment['timestamp'], comment['likes'], comment['reply_count'], comment['content'], label))
    expected_columns = ["username", "timestamp", "likes", "reply_count", "content"]
    new_df = pd.DataFrame(comments_data, columns=expected_columns)
    if os.path.exists('yorumlar.csv'):
        existing_df = pd.read_csv('yorumlar.csv')
        if list(existing_df.columns) == expected_columns:
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            existing_df.to_csv('yorumlar.csv', index=False, encoding='utf-8')
            status_label.config(text=f"{len(comments_data)} tane yorum yorumlar.csv dosyasına eklendi.")
        else:
            os.remove('yorumlar.csv')
            new_df.to_csv('yorumlar.csv', index=False, encoding='utf-8')
            status_label.config(text=f"Mevcut yorumlar.csv dosyası uyumsuz olduğu için silindi, {len(comments_data)} tane yorum tekrar yorumlar.csv olarak kaydedildi.")
    else:
        new_df.to_csv('yorumlar.csv', index=False, encoding='utf-8')
        status_label.config(text=f"{len(comments_data)} tane yorum yeni yorumlar.csv dosyasına kaydedildi.")
    run_button.config(text="Yorumları Getir", bg="#0D92F4")
    csv_button.config(state="normal", bg="#0D92F4", fg="white", padx=10, pady=5, relief="groove", borderwidth=2)
    tablo_temizle_button.config(state="normal", bg="#0D92F4", fg="white", padx=10, pady=5, relief="groove", borderwidth=2)
def csv_kaydet_btn_click(datatable, status_label):
    status_label.config(text="Kaydediliyor...")
    data = []
    for row in datatable.get_children():
        values = datatable.item(row)['values'][1:]  # ID sütunu hariç
        data.append(values)
    df = pd.DataFrame(data, columns=["username", "timestamp", "likes", "reply_count", "content", "label"])
    df.to_csv('yorumlar_analiz.csv', index=False, encoding='utf-8')
    status_label.config(text="Analiz edilmiş yorumlar 'yorumlar_analiz.csv' olarak kaydedildi.")
def yorumlari_getir_ve_analiz_et(datatable, status_label, link_entry, csv_button, stop_event):
    status_label.config(text="Yorumlar getiriliyor...")
    link = link_entry.get()
    if get_video_id(link) != None:
        comments_data = get_video_data(get_video_id(link))
        yorumlari_analiz_et_ve_tabloyu_guncelle(datatable, comments_data, status_label, csv_button, link, stop_event)
    else:
        status_label.config(text="Lütfen geçerli bir YouTube linki girin.")
def yorumlari_getir_btn_click(datatable, status_label, link_entry, csv_button, stop_event, run_button):
    if run_button["text"] == "Yorumları Getir":
        stop_event.clear()
        run_button.config(text="İptal Et", bg="#FF6F61")
        csv_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)
        tablo_temizle_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)
        threading.Thread(target=yorumlari_getir_ve_analiz_et, args=(datatable, status_label, link_entry, csv_button, stop_event), daemon=True).start()
    else:
        stop_event.set()
        run_button.config(text="Yorumları Getir", bg="#0D92F4")
        csv_button.config(state="normal", bg="#0D92F4", fg="white", padx=10, pady=5, relief="groove", borderwidth=2)
        tablo_temizle_button.config(state="normal", bg="#0D92F4", fg="white", padx=10, pady=5, relief="groove", borderwidth=2)
def tabloyu_temizle_btn_click(status_label):
    [datatable.delete(row) for row in datatable.get_children()]
    csv_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)
    tablo_temizle_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)
    status_label.config(text="Tablo başarıyla temizlendi.")

root = tk.Tk()
root.title("Yapay Sinir Ağları Yorum Analiz")
root.geometry("1280x720")

frame = tk.Frame(root)
frame.pack(pady=10)

link_entry = tk.Entry(frame, width=50)
link_entry.grid(row=0, column=0, padx=5)

stop_event = threading.Event()

run_button = tk.Button(frame, text="Yorumları Getir", command=lambda: yorumlari_getir_btn_click(datatable, status_label, link_entry, csv_button, stop_event, run_button))
run_button.grid(row=0, column=1, padx=5)
run_button.config(bg="#0D92F4", fg="white", padx=10, pady=5, relief="groove", borderwidth=2)

csv_button = tk.Button(frame, text="Tabloyu CSV Olarak Kaydet", command=lambda: csv_kaydet_btn_click(datatable, status_label))
csv_button.grid(row=0, column=2, padx=5)
csv_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)

tablo_temizle_button = tk.Button(frame, text="Tabloyu Temizle", command=lambda: tabloyu_temizle_btn_click(status_label))
tablo_temizle_button.grid(row=0, column=3, padx=5)
tablo_temizle_button.config(state="disabled", bg="#F0F0F0", fg="gray", padx=10, pady=5, relief="groove", borderwidth=2)

status_label = tk.Label(root, text="", fg="black")
status_label.pack(pady=10)

table_config = { "Video ID": 20, "Username": 100, "Date Time": 100, "Likes": 10, "Reply": 10, "Content": 750, "Label": 50 }
columns = list(table_config.keys())
datatable = ttk.Treeview(root, columns=columns, show="headings")
for col, width in table_config.items():
    datatable.heading(col, text=col)
    datatable.column(col, width=width)
datatable.pack(expand=True, fill="both")

root.mainloop()
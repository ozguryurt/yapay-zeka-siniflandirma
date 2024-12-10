import pandas as pd
df = pd.read_csv('yeni_veriler2.csv')
df = df.rename(columns={'text': 'comment'})
df['label'] = df['label'].replace({'Positive': 'pozitif', 'Negative': 'negatif', 'Notr': 'nötr'})
df = df.drop(columns=['dataset'])
df.to_csv('yeni_veriler.csv', index=False)
print("İşlemler tamamlandı ve güncellenmiş dosya kaydedildi.")
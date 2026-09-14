import pandas as pd
import pickle
import numpy as np

# Memuat model, scaler, encoder, dan rentang fitur
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    scaler_fitur = pickle.load(open('scaler_fitur.pkl', 'rb'))
    target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoder.pkl', 'rb'))
    rentang_fitur = pickle.load(open('rentang_fitur.pkl', 'rb'))
    return model, scaler_fitur, target_scaler, encoders, rentang_fitur

model, scaler_fitur, target_scaler, encoders, rentang_fitur = load_artifacts()

# Fungsi validasi numerik
def validasi_numerik(nilai, fitur):
    min_val = rentang_fitur[fitur]['min']
    max_val = rentang_fitur[fitur]['max']
    return max(min_val, min(nilai, max_val))

# Judul aplikasi Streamlit
st.title('Prediksi Harga Rumah')
st.write('Masukkan detail properti untuk memprediksi harga.')

# Input numerik menggunakan slider
in_bed = st.slider(
    'Jumlah Kamar Tidur (bedrooms)',
    min_value=int(rentang_fitur['bedrooms']['min']),
    max_value=int(rentang_fitur['bedrooms']['max']),
    value=3
)
in_bath = st.slider(
    'Jumlah Kamar Mandi (bathrooms)',
    min_value=float(rentang_fitur['bathrooms']['min']),
    max_value=float(rentang_fitur['bathrooms']['max']),
    value=2.0,
    step=0.25
)
in_sqft = st.slider(
    'Luas Ruang Keluarga (sqft_living)',
    min_value=int(rentang_fitur['sqft_living']['min']),
    max_value=int(rentang_fitur['sqft_living']['max']),
    value=2000
)
in_flr = st.slider(
    'Jumlah Lantai (floors)',
    min_value=float(rentang_fitur['floors']['min']),
    max_value=float(rentang_fitur['floors']['max']),
    value=1.0,
    step=0.5
)
in_abv = st.slider(
    'Luas di Atas Permukaan Tanah (sqft_above)',
    min_value=int(rentang_fitur['sqft_above']['min']),
    max_value=int(rentang_fitur['sqft_above']['max']),
    value=1500
)

# Input kategorikal menggunakan selectbox
list_city = list(encoders['city'].classes_)
in_city = st.selectbox(
    'Kota (city)',
    options=list_city,
    index=list_city.index('Seattle') if 'Seattle' in list_city else 0
)

list_statezip = list(encoders['statezip'].classes_)
in_zip = st.selectbox(
    'Kode Pos (statezip)',
    options=list_statezip,
    index=list_statezip.index('WA 98115') if 'WA 98115' in list_statezip else 0
)

# Ketika tombol 'Prediksi Harga' ditekan
if st.button('Prediksi Harga'):
    # Validasi dan transformasi input
    val_bed = validasi_numerik(in_bed, 'bedrooms')
    val_bath = validasi_numerik(in_bath, 'bathrooms')
    val_sqft = validasi_numerik(in_sqft, 'sqft_living')
    val_flr = validasi_numerik(in_flr, 'floors')
    val_abv = validasi_numerik(in_abv, 'sqft_above')

    enc_city = encoders['city'].transform([in_city])[0]
    enc_zip = encoders['statezip'].transform([in_zip])[0]

    # Buat DataFrame input
    x_input = pd.DataFrame({
        'bedrooms': [val_bed],
        'bathrooms': [val_bath],
        'sqft_living': [val_sqft],
        'floors': [val_flr],
        'sqft_above': [val_abv],
        'city': [enc_city],
        'statezip': [enc_zip]
    })

    # Skalakan input
    x_scaled = scaler_fitur.transform(x_input)

    # Prediksi menggunakan model
    pred_skala = model.predict(x_scaled)

    # Inverse transform untuk mendapatkan harga asli
    harga_asli = target_scaler.inverse_transform([pred_skala])

    st.success(f'### Prediksi Harga Rumah: ${harga_asli[0][0]:,.2f}')

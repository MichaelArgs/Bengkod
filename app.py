# Menampilkan judul aplikasi
st.title("Prediksi Kualitas Air")

# Deskripsi aplikasi
st.write("""
Aplikasi ini memprediksi apakah air bisa dikonsumsi (Potable) atau tidak (Not Potable)
berdasarkan beberapa parameter kualitas air. Silakan masukkan nilai-nilai berikut.
""")

# Form input untuk menerima nilai dari pengguna
ph = st.number_input("PH", min_value=0.0, max_value=14.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0, step=0.1)
solids = st.number_input("Solids", min_value=0.0, step=0.1)
chloramines = st.number_input("Chloramines", min_value=0.0, step=0.1)
sulfate = st.number_input("Sulfate", min_value=0.0, step=0.1)
conductivity = st.number_input("Conductivity", min_value=0.0, step=0.1)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, step=0.1)
turbidity = st.number_input("Turbidity", min_value=0.0, step=0.1)

# Tombol prediksi
if st.button("Prediksi Kualitas Air"):
    # Fitur input pengguna
    features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                          organic_carbon, trihalomethanes, turbidity]])

    # Memuat model yang sudah dilatih (misalnya Gradient Boosting)
    with open('gradient_boosting_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Melakukan prediksi
    prediction = model.predict(features)

    # Menampilkan hasil prediksi
    if prediction == 1:
        st.write("**Air Potable** (Dapat dikonsumsi)")
    else:
        st.write("**Air Tidak Potable** (Tidak dapat dikonsumsi)")


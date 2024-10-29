import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("price_prediction_model.pkl")

# List of all possible addresses from your training set
address_list = [
    'Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan',
       'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No',
       'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh',
       'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr',
       'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Bahar',
       'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh',
       'Elm-o-Sanat', 'Golestan', 'Shahr-e-Ziba', 'Pasdaran',
       'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi',
       'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh',
       'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari',
       'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir',
       'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat',
       'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan',
       'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook',
       'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht',
       'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr',
       'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft',
       'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad',
       'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad',
       'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj',
       'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak',
       'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr',
       'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva',
       'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri',
       'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi',
       'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad',
       'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication',
       'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan',
       'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad',
       'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River',
       'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh',
       'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh',
       'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad']

def preprocess_user_input(area, room, parking, warehouse, elevator, address):
    # 1. Convert boolean features
    boolean_features = [int(parking), int(warehouse), int(elevator)]
    
    # 2. Create a one-hot encoded dictionary for the address
    address_one_hot = {f"Address_{addr}": 0 for addr in address_list}
    if f"Address_{address}" in address_one_hot:
        address_one_hot[f"Address_{address}"] = 1  # Set the user's address to 1

    # 3. Combine all features into a single input array
    features = [area, room] + boolean_features + list(address_one_hot.values())
    return np.array(features).reshape(1, -1)  # Reshape for model input

def predict_price(area, room, parking, warehouse, elevator, address):
    input_data = preprocess_user_input(area, room, parking, warehouse, elevator, address)
    price_prediction = model.predict(input_data)
    return float(price_prediction[0])  # Return prediction as float64

# Example usage
# User inputs:
#area = 70
#room = 2
#parking = True
#warehouse = True
#elevator = False
#address = 'Shahran'

predicted_price = predict_price(area, room, parking, warehouse, elevator, address)
#print(f"Predicted Price: {predicted_price}")

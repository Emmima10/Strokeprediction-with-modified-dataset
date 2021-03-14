import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gender': 0, 'age':20, 'Hypertension':116, 'alcohol':7,'heart':1,'Residence_type':1,'avg_glucose':122,'bmi':28,'SMOKE':0})

print(r.json())
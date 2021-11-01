import datetime as dt
import streamlit as st
from funkcje import obrobka_df, load_models, podsumowanie, wykres_y, wykres_x
from PIL import Image

st.set_page_config(layout='wide')
"""
# Regresja

"""
image = Image.open('tramwaj.png')
st.image(image)
data = st.date_input('Wybierz dzień do obserwacji', value=dt.date(2021, 9, 1), min_value=dt.date(2021, 9, 1),
	                    max_value=dt.date(2021,9,30))
data_str = str(data.year) + '-'
if data.month < 10:
	data_str += '0'+str(data.month)+'-'
else:
	data_str += str(data.month)+'-'
		
if data.day < 10:
	data_str += '0'+str(data.day)
else:
	data_str += str(data.day)

byla_jazda = False

x_train, x_test, y_train, y_test, time_slices, df_t = obrobka_df('godziny.csv')

for i,j in enumerate(df_t):
	if j.split()[0] == data_str:
		byla_jazda = True
		break

if byla_jazda:
	regs = load_models(x_train, y_train)
	preds = [i.predict(x_test) for i in regs]
	"""
	Przewidywana różnica między temperaturą w łożysku przednim/tylnim, a średnią z łożysk środkowych w danym dniu września
	na podstawie modelu wytrenowanego na danych z miesięcy kwiecień-sierpień
	"""
	wykres_y(y_test, preds, data_str, df_t)

	"""
	Przebieg zmiennych na podstawie których model dokonuje estymacji
	"""
	wykres_x(x_test, df_t, data_str)
else:
	st.write("Brak przejazdów w tym dniu")

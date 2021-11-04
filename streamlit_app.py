import datetime as dt
import streamlit as st
from funkcje import obrobka_df, load_models, podsumowanie, wykres_y, wykres_x, wykres_avg, rysuj_shapy, rysuj_waterfall
from PIL import Image

st.set_page_config(layout='wide')
"""
# Regresja

"""



tramwaj = Image.open('tramwaj.png')
gait = Image.open('gait.png')
id = Image.open('id.png')
col1, col2, col3 = st.columns([1, 3, 1])
col1.title(" ")
col2.title("Rozkład wózków w tramwaju")
col3.title(" ")
col1.image(id, use_column_width=True)
col2.image(tramwaj, use_column_width=True)
col3.image(gait,use_column_width=True)
data = st.date_input('Wybierz dzień do obserwacji', value=dt.date(2021, 9, 1), min_value=dt.date(2021, 9, 1),
	                    max_value=dt.date(2021,9,30))

choice = st.radio('wyświetl dane z: ', ('regresji liniowej', 'random forrest'))

is_rf = False

if choice == 'regresji liniowej':
	is_rf = False
elif choice == 'random forrest':
	is_rf = True

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

x_train, x_test, y_train, y_test, time_slices, df_t, df_avg = obrobka_df('godziny.csv')
for i, j in enumerate(df_t):
	if j.split()[0] == data_str:
		byla_jazda = True
		break

if byla_jazda and choice is not None:
	regs = load_models(is_rf)
	preds = [i.predict(x_test) for i in regs]
	with st.expander("Wykresy Y"):
		"""
		Przewidywana różnica między temperaturą w łożysku przednim/tylnim, a średnią z łożysk środkowych w danym dniu września
		na podstawie modelu wytrenowanego na danych z miesięcy kwiecień-sierpień. Bezpieczny przedział to +- 3°C, a przedział ostrzegawczy to +- 2°C
		"""
		wykres_y(y_test, preds, data_str, df_t)

	with st.expander("Wykresy X"):
		"""
		Przebieg zmiennych na podstawie których model dokonuje estymacji
		"""
		wykres_x(x_test, df_t, data_str)
	
	with st.expander("Wykres średniej"):
		"""
		Przebieg średniej temperatury z wózków BL1-CR2, od której to różnicę przewidujemy naszym modelem
		"""
		col3, col4, col5 = st.columns([1, 3, 1])
		
		col4.pyplot(wykres_avg(df_avg, df_t, data_str))
	if not is_rf:
		"""
		Podsumowanie współczynników regresji dla każdego łożyska + wynik r^2 (miara dopasowania modelu do danych)
		"""
		st.write(podsumowanie(x_test, y_test, regs))
	else:
		rysuj_shapy(y_test, x_test, regs)
		with st.expander("Wykresy dla poszczególnych obserwacji"):
			"""
			W jaki sposób poszczególne dane wejściowe mają wpływ na predykcje modelu
			"""
			
			ilosc = time_slices[0][4] - time_slices[0][3] + 1
			n = st.slider('wybierz obserwację do analizy', 1, ilosc, 1)
			rysuj_waterfall(x_test, y_test, time_slices, regs, data_str,  n)
	st.write("metodologia opracowania modelu [link](https://docs.google.com/presentation/d/19SkSF6WnEuGmVQNwRAMcyPjQeVMGxQ3nIgM2y5akBgA/edit?usp=sharing)")
	
else:
	st.write("Brak przejazdów w tym dniu")


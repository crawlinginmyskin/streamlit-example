import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

lozyska = {
	'w1_przod_lewy': 'AL1',
	'w1_przod_prawy': 'AR1',
	'w1_tyl_lewy': 'AL2',
	'w1_tyl_prawy': 'AR2',
	'w4_przod_lewy': 'DL1',
	'w4_przod_prawy': 'DR1',
	'w4_tyl_lewy': 'DL2',
	'w4_tyl_prawy': 'DR2',
	
}


def wykres_y(y_test, preds, d, df_t, len_test=689):
	d_count = 0
	x_labels = []
	d_indices = []
	for i, j in enumerate(df_t):
		if j.split()[0] == d:
			d_count += 1
			x_labels.append(j)
			d_indices.append(i)
	x = [i for i in range(d_count)]
	preds = np.array(preds)
	fig = plt.figure(figsize=(40, 45))
	for c, n in enumerate(y_test.columns):
		predykcja = preds[c, d_indices[0]-len_test:d_indices[-1]-len_test+1]
		plt.subplot(8, 2, 1+c)
		plt.title(lozyska[n] + ' w dniu ' + d)
		plt.plot(x, y_test.loc[d_indices[0]:d_indices[-1], n], 'ro-', label="pomiary rzeczywiste")
		plt.plot(x, predykcja, 'o-', label="predykcja")
		plt.plot(x, [i + 3 for i in predykcja], 'go--', label='bezpieczny przedział')
		plt.plot(x, [i - 3 for i in predykcja], 'go--')
		plt.legend()
		plt.ylabel('różnica między ' + lozyska[n] + ' a średnią BR1-CL2', fontsize=14)
		plt.fill_between(x, [i + 3 for i in predykcja], [i - 3 for i in predykcja], alpha=0.3, color='green')
		plt.xticks(x, df_t[d_indices[0]:d_indices[-1] + 1])
	st.pyplot(fig)
	return None


def wykres_x(x_test, df_t, d):
	d_count = 0
	x_labels = []
	d_indices = []
	for i, j in enumerate(df_t):
		if j.split()[0] == d:
			d_count += 1
			x_labels.append(j)
			d_indices.append(i)
	x = [i for i in range(d_count)]
	fig = plt.figure(figsize=(40, 20))
	for c, n in enumerate(x_test.columns):
		if n != 'kierunekB':
			if c == 0:
				plt.subplot(2, 2, 1)
			else:
				plt.subplot(2, 2, c)
			plt.plot(x, x_test.loc[d_indices[0]:d_indices[-1], n], 'o-', label=n)
			plt.xticks(x, df_t[d_indices[0]:d_indices[-1] + 1])
			plt.legend()
			plt.title(n + ' w dniu ' + d)
	st.pyplot(fig)
	return None


def obrobka_df(filename):
	# doprowadzenie df do plottable wersji
	df = pd.read_csv(filename, index_col=0)
	for i, j in enumerate(df['avg_max_2_3']):
		df.at[i, 'w1_przod_lewy'] -= j
		df.at[i, 'w1_przod_prawy'] -= j
		df.at[i, 'w1_tyl_lewy'] -= j
		df.at[i, 'w1_tyl_prawy'] -= j
		df.at[i, 'w4_przod_lewy'] -= j
		df.at[i, 'w4_przod_prawy'] -= j
		df.at[i, 'w4_tyl_lewy'] -= j
		df.at[i, 'w4_tyl_prawy'] -= j
	
	df = df.drop(['avg_max_2_3', 'kierunekA', 'max_temp_silnika'], axis=1)
	df = df[df['przebieg'] > 5].reset_index(drop=True)
	df = df.dropna()
	to_drop = []
	for i, j in enumerate(df['temp_zew']):
		if j == 0.0:
			to_drop.append(i)
	
	df = df.drop(to_drop).reset_index(drop=True)
	train_test = []
	b = False
	for i, j in enumerate(df['Data_czas']):
		if j.split()[0] == '2021-09-01':
			b = True
		if b:
			train_test.append('test')
		else:
			train_test.append('train')
	df['train_test'] = train_test
	df_x = df[['temp_zew', 'kierunekB', 'przebieg', 'max_temp_przekladnie', 'max_predkosc_osi', 'train_test']]
	df_y = df[df.columns[1:9]]
	df_y['train_test'] = df_x['train_test'].to_numpy()
	df_t = df['Data_czas']
	x_train = df_x[df_x['train_test'] == 'train'].drop(['train_test'], axis=1)
	x_test = df_x[df_x['train_test'] == 'test'].drop(['train_test'], axis=1)
	y_train = df_y[df_y['train_test'] == 'train'].drop(['train_test'], axis=1)
	y_test = df_y[df_y['train_test'] == 'test'].drop(['train_test'], axis=1)
	
	# generacja time-slice'ów
	
	test_time = []
	for i, j in enumerate(df_t):
		if i > len(y_train['w1_przod_lewy']):
			test_time.append(j)
	test_time = [test_time[i].split()[0] for i in range(len(test_time))]
	time_slices = []
	check = test_time[0]
	start = len(y_train['w1_przod_lewy'])
	length = 0
	for i, j in enumerate(test_time):
		if j != check:
			time_slices.append([check, start, start + length, i - length, i])
			check = j
			start = start + length + 1
			length = 0
		else:
			length += 1
	
	return x_train, x_test, y_train, y_test, time_slices, df_t


def load_models(x_train, y_train):
	return [LinearRegression().fit(x_train, y_train[i]) for i in y_train.columns]


def podsumowanie(x_test, y_test, regs):
	kolumny = ['łożyzko']
	for i, j in enumerate(x_test.columns.values):
		# print(regs[c].coef_[i], j)
		kolumny.append(j)
	
	kolumny.append('r^2')
	podsumowanie = pd.DataFrame(columns=kolumny)
	
	for c, n in enumerate(y_test.columns):
		row = [n]
		for i, j in enumerate(x_test.columns.values):
			row.append(round(regs[c].coef_[i], 3))
		
		row.append(round(regs[c].score(x_test, y_test[n]), 3))
		podsumowanie = podsumowanie.append(pd.DataFrame([row], columns=kolumny))
	podsumowanie = podsumowanie.reset_index(drop=True)
	return podsumowanie

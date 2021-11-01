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
	'w4_tyl_prawy': 'DR2'
}
indeksy = {
	'w1_przod_lewy': 5,
	'w1_przod_prawy': 1,
	'w1_tyl_lewy': 6,
	'w1_tyl_prawy': 2,
	'w4_przod_lewy': 7,
	'w4_przod_prawy': 3,
	'w4_tyl_lewy': 8,
	'w4_tyl_prawy': 4
}

ylim = {
	'przebieg': 50,
	'max_temp_przekladnie': 90,
	'max_predkosc_osi': 80
}


def wykres_y(y_test, preds, d, df_t, len_test=689):
	d_count = 0
	x_labels = []
	d_indices = []
	for i, j in enumerate(df_t):
		if j.split()[0] == d:
			d_count += 1
			x_labels.append(int(j.split()[1].split(':')[0]))
			d_indices.append(i)
	x = [i for i in range(24)]
	preds = np.array(preds)
	
	fig = plt.figure(figsize=(30, 45))
	for c, n in enumerate(y_test.columns):
		predykcja = preds[c, d_indices[0]-len_test:d_indices[-1]-len_test+1]
		plt.subplot(8, 4, indeksy[n])
		plt.title(lozyska[n] + ' w dniu ' + d)
		#plt.plot(x, x, alpha=0)
		plt.xlim(0, 24)
		plt.ylim(np.min(y_test.to_numpy())-3, np.max(y_test.to_numpy())+3)
		for i in range(d_count):
			if i == 0:
				plt.plot([x_labels[i], x_labels[i+1]], [y_test.loc[d_indices[i], n], y_test.loc[d_indices[i + 1], n]]
				, 'ro-', label="pomiary rzeczywiste")
				plt.plot([x_labels[i], x_labels[i+1]], [predykcja[i], predykcja[i+1]], 'bo-', label='predykcja')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i]+3, predykcja[i + 1]+3], 'go-', label='bezpieczny przedzial')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i] - 3, predykcja[i + 1] - 3], 'go-')
			elif i != d_count-1:
				plt.plot([x_labels[i], x_labels[i+1]], [y_test.loc[d_indices[i], n], y_test.loc[d_indices[i + 1], n]]
				, 'ro-')
				plt.plot([x_labels[i], x_labels[i+1]], [predykcja[i], predykcja[i+1]], 'bo-')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i]+3, predykcja[i + 1]+3], 'go-')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i] - 3, predykcja[i + 1] - 3], 'go-')

		plt.legend()
		plt.ylabel('różnica między ' + lozyska[n] + ' a średnią BR1-CL2', fontsize=12)
		plt.fill_between(x_labels, [i + 3 for i in predykcja], [i - 3 for i in predykcja], alpha=0.2, color='green')
		plt.xticks(x, [str(i) + ':00' for i in x], rotation=45)
	st.pyplot(fig)
	return None


def wykres_x(x_test, df_t, d):
	d_count = 0
	x_labels = []
	d_indices = []
	for i, j in enumerate(df_t):
		if j.split()[0] == d:
			d_count += 1
			x_labels.append(int(j.split()[1].split(':')[0]))
			d_indices.append(i)
	x = [i for i in range(24)]

	fig = plt.figure(figsize=(25, 10))
	for c, n in enumerate(x_test.columns):
		if n != 'kierunekB':
			if c == 0:
				plt.subplot(2, 2, 1)
				plt.ylim(-10, 30)
			else:
				plt.subplot(2, 2, c)
				plt.ylim(0, ylim[n])
			
			for i in range(d_count):
				if i != d_count -1:
					plt.plot([x_labels[i], x_labels[i+1]], [x_test.loc[d_indices[i], n], x_test.loc[d_indices[i+1], n]], 'o-', c='#1f77b4')
			plt.xticks(x, [str(i) + ':00' for i in x], rotation=45)
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

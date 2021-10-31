import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


def wykres1(y_test, preds, c, d, r):
	x = list(range(len(y_test)))
	fig = plt.figure(figsize=(20, 6))
	plt.plot(x, y_test, 'ro-', label="pomiary rzeczywiste")
	
	plt.plot(x, preds, 'o-', label="predykcja")
	
	plt.plot(x, [i + 3 for i in preds], 'go--', label='bezpieczny przedzial')
	plt.plot(x, [i - 3 for i in preds], 'go--')
	plt.legend()
	plt.title(c + ' ' + r + ' w dniu ' + d)
	st.pyplot(fig)
	return None


def wykres2(y_test, preds, c, d, r):
	x = list(range(len(y_test)))
	y_p = y_test.reset_index()
	fig = plt.figure(figsize=(20, 6))
	plt.scatter(x, y_test, label="pomiary rzeczywiste")
	
	plt.scatter(x, preds, label="predykcja")
	for i in x:
		plt.plot((i, i), (preds[i], y_p.at[i, c]), c='black')
	plt.legend()
	plt.title(c + ' ' + r + ' w dniu ' + d)
	st.pyplot(fig)
	return None


def obrobka_df(filename):
	# doprowadzenie df do plottable wersji
	df = pd.read_csv(filename, index_col=0)
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
	
	return x_train, x_test, y_train, y_test, time_slices


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


def rysuj(y_test, preds, d, time_slices):
	for c, n in enumerate(y_test.columns):
		for i, j in enumerate(time_slices):
			if j[0] == d:
				wykres1(y_test.loc[j[1]:j[2], n], preds[c][j[3]:j[4] + 1], n, j[0], 'regresja liniowa')
				wykres2(y_test.loc[j[1]:j[2], n], preds[c][j[3]:j[4] + 1], n, j[0], 'regresja liniowa')

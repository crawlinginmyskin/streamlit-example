import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import joblib
from joblib import wrap_non_picklable_objects, delayed

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


tytuly = {
	'przebieg': "Przebieg w danej godzinie",
	'max_temp_przekladnie': "Maksymalna temperatura przekładni w danej godzinie",
	'max_predkosc_osi': "Maksymalna prędkosć w danej godzinie",
	'temp_zew': 'Temperatura zewnętrzna',
	'kierunekB': 'Kierunek w którym jechał tramwaj'
}

jednostki = {
	'przebieg': 'km',
	'max_temp_przekladnie': "°C",
	'max_predkosc_osi': "km/h",
	'temp_zew': '°C'
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
		plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
		for i in range(d_count):
			if i == 0:
				plt.plot([x_labels[i], x_labels[i+1]], [y_test.loc[d_indices[i], n], y_test.loc[d_indices[i + 1], n]]
				,'bo-', linewidth=3, label="pomiary rzeczywiste")
				plt.plot([x_labels[i], x_labels[i+1]], [predykcja[i], predykcja[i+1]], '-.', color='black',  label='predykcja')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i]+3, predykcja[i + 1]+3], 'g-', label='przedział bezpieczny', linewidth=0)
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i] - 3, predykcja[i + 1] - 3], 'g-', linewidth=0)
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i] + 5, predykcja[i + 1] + 5], color='orange', linewidth=0, label='przedział ostrzegawczy')
			elif i != d_count-1:
				plt.plot([x_labels[i], x_labels[i+1]], [y_test.loc[d_indices[i], n], y_test.loc[d_indices[i + 1], n]]
				, 'bo-', linewidth=3)
				plt.plot([x_labels[i], x_labels[i+1]], [predykcja[i], predykcja[i+1]], '-.', color='black')
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i]+3, predykcja[i + 1]+3], 'g', linewidth=0)
				plt.plot([x_labels[i], x_labels[i + 1]], [predykcja[i] - 3, predykcja[i + 1] - 3], 'g', linewidth=0)
		
		leg = plt.legend()
		for line in leg.get_lines():
			line.set_linewidth(2.0)
		plt.ylabel('różnica między ' + lozyska[n] + ' a średnią BR1-CL2', fontsize=12)
		plt.fill_between(x_labels, [i + 3 for i in predykcja], [i - 3 for i in predykcja], alpha=0.2, color='green')
		plt.fill_between(x_labels, [i + 3 for i in predykcja], [i + 5 for i in predykcja], alpha=0.2, color='orange')
		plt.fill_between(x_labels, [i - 3 for i in predykcja], [i - 5 for i in predykcja], alpha=0.2, color='orange')
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

	fig = plt.figure(figsize=(30, 5))
	for c, n in enumerate(x_test.columns):
		if n != 'kierunekB':
			if c == 0:
				plt.subplot(1, 5, 1)
				plt.ylim(-10, 30)
			else:
				plt.subplot(1, 5, c)
				plt.ylim(0, ylim[n])
			
			for i in range(d_count):
				if i != d_count - 1:
					plt.plot([x_labels[i], x_labels[i+1]], [x_test.loc[d_indices[i], n], x_test.loc[d_indices[i+1], n]], 'o-', c='#1f77b4')
			plt.xticks(x, [str(i) + ':00' for i in x], rotation=45)
			plt.ylabel(jednostki[n])
			plt.title(tytuly[n])
		else:
			plt.subplot(1,5,5)
			plt.ylim(-1.5, 1.5)
			plt.title(tytuly[n])
			to_plot = []
			for i in range(d_count):
				if x_test.loc[d_indices[i], n] == 0:
					to_plot.append(1)
				else:
					to_plot.append(-1)
			for i in range(d_count):
				if i != d_count - 1:
					plt.plot([x_labels[i], x_labels[i+1]], [to_plot[i], to_plot[i+1]], c='#1f77b4')
			
			for i in range(d_count):
				plt.plot([x_labels[i], x_labels[i]],  [to_plot[i], 0], c='black')
				plt.scatter(x_labels[i], to_plot[i], c='black')
			
			plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
			plt.xticks(x, [str(i) + ':00' for i in x], rotation=45)
			plt.yticks([-1, 0, 1], ['wózek A\nz tyłu', '0', 'wózek A\nz przodu'], rotation=45)
	st.pyplot(fig)
	return None


def wykres_avg(df_avg, df_t, d, len_test=689):
	d_count = 0
	x_labels = []
	d_indices = []
	for i, j in enumerate(df_t):
		if j.split()[0] == d:
			d_count += 1
			x_labels.append(int(j.split()[1].split(':')[0]))
			d_indices.append(i)
	x = [i for i in range(24)]
	fig = plt.figure(figsize=(15, 3))
	plt.ylim(0, 70)
	for i in range(0, 70, 10):
		plt.axhline(y=i, color='k', linestyle='--', alpha=0.5)
	for i in range(d_count):
		if i != d_count - 1:
			plt.plot([x_labels[i], x_labels[i+1]], [df_avg.loc[i+len_test, 'avg_max_2_3'], df_avg.loc[i+len_test+1, 'avg_max_2_3']], 'o-',c='#1f77b4')
	plt.xticks(x, [str(i) + ':00' for i in x], rotation=45)
	plt.ylabel('°C')
	plt.title('Przebieg średnich temperatur z wózków 2 i 3')
	#st.pyplot(fig)
	return fig


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
	df_avg = df[['Data_czas', 'avg_max_2_3', 'przebieg']]
	df = df[df['przebieg'] > 5].reset_index(drop=True)
	df = df.drop(['avg_max_2_3', 'kierunekA', 'max_temp_silnika'], axis=1)
	df_avg = df_avg[df_avg['przebieg'] > 5].reset_index(drop=True)
	df_avg = df_avg.drop(['przebieg'], axis=1)
	
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
	
	return x_train, x_test, y_train, y_test, time_slices, df_t, df_avg


def load_models(is_rf):
	test = []
	if is_rf:
		for i in range(8):
			test.append(pickle.load(open('rf_' + str(i) + '.sav', 'rb')))
	else:
		for i in range(8):
			test.append(pickle.load(open(str(i) + '.sav', 'rb')))
	return test


def podsumowanie(x_test, y_test, regs):
	kolumny = ['łożysko']
	for i, j in enumerate(x_test.columns.values):
		kolumny.append(tytuly[j])
	
	kolumny.append('r^2')
	podsumowanie = pd.DataFrame(columns=kolumny)
	
	for c, n in enumerate(y_test.columns):
		row = [lozyska[n]]
		for i, j in enumerate(x_test.columns.values):
			row.append(round(regs[c].coef_[i], 3))
		
		row.append(round(regs[c].score(x_test, y_test[n]), 3))
		podsumowanie = podsumowanie.append(pd.DataFrame([row], columns=kolumny))
	podsumowanie = podsumowanie.reset_index(drop=True)
	return podsumowanie


def rysuj_shapy(y_test, x_test, rf_regs):
	fig = plt.figure(figsize=(50, 60))
	x_t = x_test.copy()
	x_t = x_t.rename(columns={'kierunekB': 'kierunek_A_z_przodu'})
	for c, n in enumerate(y_test.columns):
		plt.subplot(8, 4, indeksy[n])
		explainer = shap.TreeExplainer(rf_regs[c], x_t)
		shap_values = explainer(x_t, check_additivity=False)
		shap.initjs()
		shap.plots.beeswarm(shap_values, plot_size=None, show=False)
		plt.title(lozyska[n])
		
	st.pyplot(fig)
	return None


@wrap_non_picklable_objects
def rysuj_waterfall(x_test, y_test, time_slices, regs, data_str, n):
	day_start_end = [0, 0, 0, 0]
	c1,c2,c3,c4 = st.columns([1,1,1,1])
	c5,c6,c7,c8 = st.columns([1,1,1,1])
	kolumny = [c1,c2,c3,c4,c5,c6,c7,c8]
	x_t = x_test.copy()
	x_t = x_t.rename(columns={'kierunekB': 'kierunek_A_z_przodu'})
	for i, j in enumerate(time_slices):
		if j[0] == data_str:
			day_start_end = time_slices[i][1:]
	for c, d in enumerate(y_test.columns):
		fig = plt.figure(figsize=(1, 1))
		explainer = shap.TreeExplainer(regs[c], x_t)
		shap_values = explainer(x_t, check_additivity=False)
		shap.initjs()
		shap.plots.waterfall(shap_values[day_start_end[2] + n - 1])
		plt.title(lozyska[d])
		kolumny[indeksy[d]-1].pyplot(fig)
		
	return None


def rysuj_barplot(x_test, y_test, regs):
	c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
	c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
	kolumny = [c1, c2, c3, c4, c5, c6, c7, c8]
	x_t = x_test.copy()
	x_t = x_t.rename(columns={'kierunekB': 'kierunek_A_z_przodu'})
	for c, d in enumerate(y_test.columns):
		fig = plt.figure(figsize=(1, 1))
		explainer = shap.TreeExplainer(regs[c], x_t)
		shap_values = explainer(x_t, check_additivity=False)
		shap.initjs()
		shap.plots.bar(shap_values)
		plt.title(lozyska[d])
		kolumny[indeksy[d] - 1].pyplot(fig)
	return None

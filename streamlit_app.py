from collections import namedtuple
import altair as alt
import math
import pandas as pd
import datetime as dt
import streamlit as st
import importlib
import matplotlib.pyplot as plt
from funkcje import obrobka_df, load_models,podsumowanie, rysuj

"""
# Regresja

"""


with st.echo(code_location='below'):
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
	st.write(data_str)
	#x_train, x_test, y_train, y_test, time_slices = obrobka_df('C:\\Users\\FilipZiętara\\Desktop\\dashboard\\streamlit-example\\godziny.csv')
	x_train, x_test, y_train, y_test, time_slices = obrobka_df('godziny.csv')
	regs = load_models(x_train, y_train)
	st.write(podsumowanie(x_test, y_test, regs))
	preds = [i.predict(x_test) for i in regs]
	rysuj(y_test, preds, data_str, time_slices)

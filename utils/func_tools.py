def fillna_col_anterior(data_csv, nandzero = True):
    import numpy as np
    import pandas as pd
    if nandzero :
        data_csv = data_csv.replace(0.0, None)
    data_csv = data_csv.fillna(method='bfill',axis=0)
    return data_csv

def fillna_2013(data_csv_localidad, estra, data_csv_zona):
    import numpy as np
    import pandas as pd

    # Identificar número de zonas
    n = estra['Zona'].unique()
    zonas = np.sort(n[~pd.isna(n)])
    
    detalle = []
    for nombre_zona in zonas:
        indexl = estra.index[estra.Zona == nombre_zona]
        detalle.append([nombre_zona,indexl])

    detalle_total = []
    for nombre_z, indxs in detalle:
        for ind in indxs:
            detalle_total.append([nombre_z,ind,data_csv_localidad['ID Localidad'].index[data_csv_localidad['ID Localidad']==ind]])

    for x,linea in enumerate(detalle_total):
        datos_anos_z = data_csv_zona[data_csv_zona.Zona == linea[0]]
        for year in range(datos_anos_z.shape[0]):
            # Alinear la mismo año
            datos_anos_l = data_csv_localidad.iloc[linea[2]]
            if year == 0:
                # sacar los porcentajes de participación por localidad, zona y mes en el 2012
                per2012 = datos_anos_l.iloc[year,2:-1].values/datos_anos_z.iloc[year,2:-1].values
            if year == 1:
                # fila en el dataset de localidades original correspondiente al 2013
                fila = linea[2].values[1]
                # sacar el número de residuos en base al porcentaje del 2012 sobre los totales por zona del 2013
                datatofill = datos_anos_z.iloc[year,2:-1].values*per2012
                # llenar el dataset de localidades original
                data_csv_localidad.iloc[fila,2:-1] = datatofill

    # llenar los meses correspondientes a noviembre y diciembre con el correspondiente al año anterior como se hizo con zonas
    data_csv_localidad = data_csv_localidad.fillna(method='ffill',axis=0)
    return data_csv_localidad  

def fillna(data_csv, nandzero = True):
    import numpy as np
    import pandas as pd
    if nandzero :
        data_csv = data_csv.replace(0.0, None)
    data_csv_v = data_csv.values 
    for r in range(data_csv_v.shape[0]):
        try:
            control = np.isnan(data_csv_v[r,2:-1]).all()
        except:
            control = False
        if control:
            if r > 0 and r < data_csv_v.shape[0]:
                media_inter = (data_csv_v[r-1,2:-1]+data_csv_v[r+1,2:-1])/2
                data_csv.iloc[r,2:-1] = media_inter
    #data_csv.iloc[:,2:-3] = data_csv.iloc[:,2:-3].interpolate(method='quadratic')
    data_csv = data_csv.fillna(method='bfill',axis=0)
    return data_csv

def locs_porciento_anos(estra, data_csv_localidad, data_csv_zona):
    import numpy as np
    import pandas as pd
    
    # Identificar número de zonas
    n = estra['Zona'].unique()
    zonas = np.sort(n[~pd.isna(n)])
    detalle = []
    
    for nombre_zona in zonas:
        indexl = estra.index[estra.Zona == nombre_zona]
        detalle.append([nombre_zona,indexl])
    
    detalle_total = []
    for nombre_z, indxs in detalle:
        for ind in indxs:
            detalle_total.append([nombre_z,ind,data_csv_localidad['ID Localidad'].index[data_csv_localidad['ID Localidad']==ind]])
    
    dic_localidades_per = {}
    for x,linea in enumerate(detalle_total):
        dic_localidades_per[linea[1]] = {}
        datos_anos_z = data_csv_zona[data_csv_zona.Zona == linea[0]]
        #print(datos_anos_z)
        for r in range(datos_anos_z.shape[0]):
            dic_localidades_per[linea[1]]['201'+str(r+2)] = []
            # Alinear la mismo año
            datos_anos_l = data_csv_localidad.iloc[linea[2]]
            for c in range(datos_anos_z.shape[1]):
                if c>=2:
                    porcentaje = (datos_anos_l.iloc[r,c])/datos_anos_z.iloc[r,c]
                    dic_localidades_per[linea[1]]['201'+str(r+2)].append(porcentaje)
                    
    return dic_localidades_per


def plot_series_zona(Archivos_csvs, suavizado = 1, metodo='dinamico'):
    # Librerias a usar
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from bokeh.io import show
    from bokeh.layouts import row
    from bokeh.palettes import Viridis3
    from bokeh.plotting import figure
    from bokeh.models import CheckboxGroup, CustomJS, RadioGroup
    from matplotlib import colors as mcolors
    #-
    estra = Archivos_csvs[0]
    data_csv = Archivos_csvs[1]
    data_csv_copy = data_csv.copy()
    # Solo dejar valores de las series 
    data_csv_copy = data_csv_copy.drop(columns = ['AÑO','Población Por localidad'])
    # Eliminar espacios en blanco
    data_csv_copy['Zona'] = data_csv_copy['Zona'].str.strip()
    # Identificar número de zonas
    n = estra['Zona'].unique()
    zonas = np.sort(n[~pd.isna(n)])
    # Crear periodos de tiempo, longitud de las series
    years = np.sort(data_csv['AÑO'].unique())
    period = len(years)*12
    rng = pd.date_range('1/1/'+str(int(years[0])), periods=period, freq='M')
    # crear las series de datos y almacenarlas en un arreglo
    dict_series = {}
    series_list = []
    for ind, zona in enumerate(zonas):
        l = data_csv_copy[data_csv_copy.Zona == zona].values
        result = []
        for r in range(l.shape[0]):
            result = np.concatenate([result, np.asfarray(l[r,1:],float)])
        # Arreglar la series
        s = pd.Series(result, dtype=float, index=rng)
        # Llenar valores en 0
        s.replace(to_replace = 0, method='ffill', inplace=True)
        # Suavizar la serie
        s = s.rolling(suavizado).mean()
        dict_series[zona] = s
        series_list.append([zona,s])
        
    if metodo == 'estatico':
        p = pd.DataFrame(dict_series)
        pp = p.plot(figsize=(10,10),subplots=True,  title= 'Residuos recogidos (toneladas) por zona anualmente')
        #pp.set_xlabel("Años")
        #pp.set_xlabel("Residuos recogidos (Toneladas) ")
    elif metodo == 'dinamico':
        sorted_names = ['chocolate', 'cornflowerblue', 'black', 'brown', 'blue', 'blueviolet', 'burlywood', 'cadetblue', 'chartreuse', 'coral', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
        #-
        #output_file("'Residuos_recogidos_zona_plots'.html", title="'Residuos recogidos (toneladas) por zona anualmente' gráfica")
        TOOLS = 'crosshair,save,pan,box_zoom,reset,wheel_zoom'
        p = figure(title='Residuos recogidos (toneladas) por zona anualmente', y_axis_type="linear",x_axis_type='datetime', tools = TOOLS)
        lineas = []
        for i,s in enumerate(series_list):
            l = p.line(rng, s[1], legend=s[0], line_color=sorted_names[i], line_width = 3, line_alpha=0.8)
            lineas.append(l)
            
        #checkbox = RadioGroup(
        #                labels=list(zonas), active=0, width=100)
        checkbox = CheckboxGroup(labels=list(zonas),
                         active=[0,1,2,3,4,5], width=100)
        checkbox.callback = CustomJS.from_coffeescript(args=dict(l0=lineas[0], l1=lineas[1], l2=lineas[2], l3=lineas[3], 
                                                                 l4=lineas[4], l5=lineas[5], checkbox=checkbox), code="""   
        
        l0.visible = 0 in checkbox.active;
        l1.visible = 1 in checkbox.active;
        l2.visible = 2 in checkbox.active;
        l3.visible = 3 in checkbox.active;
        l4.visible = 4 in checkbox.active;
        l5.visible = 5 in checkbox.active;
        """)

        p.xaxis.axis_label = 'Año'
        p.yaxis.axis_label = 'Residuos recogidos (Toneladas)'

        layout = row(checkbox, p)
        show(layout)
    return layout, series_list
        
def figure_bokeh_bar(index,columns,colors,data,ano):
    from bokeh.core.properties import value
    from bokeh.io import show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool
    
    p = figure(x_range=index, plot_height=350,title='Participación localidad en los totales en '+ str(ano),
                 toolbar_location=None)
    
    #output_file("porcentaje"+str(ano)+".html")
    source = ColumnDataSource(data=data)
    p.vbar_stack(columns, x='Meses', width=0.5, color=colors, source=data,
                 legend=[value(x) for x in columns])

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_right"
    p.legend.orientation = "vertical"
    p.xaxis.axis_label = 'Años'
    p.yaxis.axis_label = 'Porcentaje participación'

    renderers = p.vbar_stack(columns, x='Meses', width=0.5, color=colors, source=source,
                             legend=[value(x) for x in columns], name=columns)

    for r in renderers:
        column = r.name
        hover = HoverTool(tooltips=[
            ("%s total" % column, "@%s" % column)
        ], renderers=[r])
        p.add_tools(hover)
    

    return p

def strip_accents(text):
    import re
    import unicodedata
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    import re
    import unicodedata
    """
    Convert input text to id.
    :param text: The input string.
    :type text: String.
    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    text = re.sub('[ ]+', '_', text)
    text = re.sub('[^0-9a-zA-Z_-]', '', text)
    return text
        
def plot_participacion_localidad(Archivos_csvs, metodo='dinamico'):
    # Librerias a usar
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from bokeh.core.properties import value
    from bokeh.io import show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.layouts import row
    from bokeh.layouts import gridplot
    from bokeh.models import Legend
    
    estra = Archivos_csvs[0]
    data_csv_localidad = Archivos_csvs[1]
    data_csv_zona = Archivos_csvs[2]
    
    participacion_porcentual_l = locs_porciento_anos(estra, data_csv_localidad, data_csv_zona)
    results = []
    if metodo == 'estatico':
        # Eliminar espacios en blanco
        data_csv_zona['Zona'] = data_csv_zona['Zona'].str.strip()
        # Identificar número de zonas
        n = estra['Zona'].unique()
        zonas = np.sort(n[~pd.isna(n)])
        # Crear periodos de tiempo, longitud de las series
        years = np.sort(data_csv_zona['AÑO'].unique())
        # Crear periodos de tiempo, longitud de las series
        index = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre', 'Población']
        for ano in years:
            detalle = []
            for nombre_zona in zonas:
                indexl = estra.index[estra.Zona == nombre_zona]
                detalle.append([nombre_zona,indexl])

            for nombre_zona, zonaloc_val in detalle:
                lista_temporal = []
                lista_temporal_id = []
                for ind in zonaloc_val:
                    dict_anos_l = participacion_porcentual_l[ind]
                    lista_temporal.append(np.array(dict_anos_l[str(ano)]))
                    lista_temporal_id.append(ind)
                dt = []
                for h in lista_temporal:
                    dt = np.concatenate((dt,h), axis=None)
                datadt = np.array(dt).reshape((len(h),len(lista_temporal)), order='F')
                df2 = pd.DataFrame(datadt, columns=[estra['Nombre Localidad'][estra.index == loi].values[0] for loi in lista_temporal_id], index= index)
                axes = df2.plot.bar(figsize=(8,8),stacked=True, title= 'Año ' + str(ano)+' - ' +estra['Zona'][estra.index == lista_temporal_id[0]].values[0]);
                axes.legend(bbox_to_anchor=(1.2, 1.0))
                fig = axes.get_figure()
                plt.show(block=False)
                plt.close(fig)
                
    elif metodo == 'dinamico':
        # Eliminar espacios en blanco
        data_csv_zona['Zona'] = data_csv_zona['Zona'].str.strip()
        # Identificar número de zonas
        n = estra['Zona'].unique()
        zonas = np.sort(n[~pd.isna(n)])
        # Crear periodos de tiempo, longitud de las series
        years = np.sort(data_csv_zona['AÑO'].unique())
        # Crear periodos de tiempo, longitud de las series
        index = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Sep', 'Oct', 'Nov', 'Dic', 'Población']
        dicts = {}
        
        for ano in years:
            colors_t = ["#c9d9d3", "#718dbf", "#e84d60","#fdae6b", "#addd8e", "#8856a7"]
            detalle = []
            dicts[ano] = []
            for nombre_zona in zonas:
                indexl = estra.index[estra.Zona == nombre_zona]
                detalle.append([nombre_zona,indexl])
            # Detalles de la zona, la localidad y que indices tiene esa localidad 
            for nombre_zona, zonaloc_val in detalle:
                lista_temporal = []
                lista_temporal_id = []
                for ind in zonaloc_val:
                    dict_anos_l = participacion_porcentual_l[ind]
                    lista_temporal.append(np.array(dict_anos_l[str(ano)]))
                    lista_temporal_id.append(ind)
                dt = []
                # se guarda y se reescala la informacion en una matriz de localidades que pertenecen a esa zona y sus porcentajes en el año
                for h in lista_temporal:
                    dt = np.concatenate((dt,h), axis=None)
                datadt = np.array(dt).reshape((len(h),len(lista_temporal)), order='F')
                # nombres de las localidades para que participan en cada zona y cada añoplotear
                columns=[text_to_id(estra['Nombre Localidad'][estra.index == loi].values[0]) for loi in lista_temporal_id]
                
                colors = []
                data = {}
                data['Meses'] = index
                # Numero de colores de acuerdo al numero de localidades presentes 
                for num, cl in enumerate(columns):
                    data[cl] = datadt[:,num].tolist()
                    colors.append(colors_t[num])
                # almacenar la información por cada año y zona en un diccionario con lo necesario para plotear
                dicts[ano].append([index, columns, colors, data, ano])  
        for y,dy in enumerate(years):
            ps = []
            # crear las figuras
            for index, columns, colors, data, ano in dicts[dy]: 
                p = figure_bokeh_bar(index,columns,colors,data,ano)
                p.legend.orientation = "horizontal"
                legend = Legend(location=(-10,0))
                p.add_layout(legend, 'right')
                ps.append(p)
                
            # crear la grilla
            grid= gridplot([[ps[0],ps[1]],[ps[2],ps[3]],[ps[4],ps[5]]])
            # mostrar los resultados
            #results.append(grid)
            show(grid)
            #show(row(ps[0],ps[1],ps[2],ps[3],ps[4],ps[5]))
            
def plot_series_localidad(data_localidad_pd, estra):
    import holoviews as hv
    hv.extension('bokeh')
    data_csv_localidad_copy = data_localidad_pd.copy()
    for ind,id_ in enumerate(data_localidad_pd['ID Localidad'].values):
        localidad = estra['Nombre Localidad'][estra.index == id_].values[0]
        data_csv_localidad_copy.iloc[ind,0] = localidad
    vdims = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 
            'Octubre', 'Noviembre', 'Diciembre', 'Población Por localidad']
    ds = hv.Dataset(data_csv_localidad_copy, ['ID Localidad','AÑO'], vdims)
    return ds


def red_lstm_corto_plazo(data_csv_np, epoch=200 ,porcentaje_entrenamiento=0.90):
    import numpy
    import matplotlib.pyplot as plt
    import pandas
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    
    # Cuadrar la semilla para poder reproducir en las mismas condiciones
    numpy.random.seed(7)
    dataset = data_csv_np.astype('float32')
    dataset_copy = dataset.copy()
    # Dividir para train y split
    train_size = int(len(dataset) * porcentaje_entrenamiento) + 1 # agrego un nuevo valor en train
    test_size = len(dataset)+2 - train_size # Agrego uno en train y otro en test
    #-----------------------------------------------------------
    # Cuadrar los puntos para que se mostrar en la gráfica correctamente y no perder datos
    last_row = numpy.array(dataset.iloc[-1,:].values).reshape(1,1)
    last_train_row = numpy.array(dataset.iloc[train_size-1,:].values).reshape(1,1)
    dataset_split = dataset.iloc[:train_size-1,:]
    dataset_split = dataset_split.append(pandas.DataFrame(last_train_row))
    dataset_split2 = dataset.iloc[train_size-1:,:]
    dataset_split2 = dataset_split2.append(pandas.DataFrame(last_row))
    frames = [dataset_split, dataset_split2]
    dataset = pandas.concat(frames)
    #------------------------------------------------------------
    # Normalizar el dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset_copy = scaler.fit_transform(dataset_copy)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print('Valores en train: ',len(train)-1, 'Valores en test: ',len(test)-1)
    # Covertir un array de valores en una matrix dataset
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    
    # redimensionar a X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #print(testX, testY)
    # redimensionar la entrada para que sea de la forma: [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #print(trainX.shape)
    #print(testX.shape)
    # crear y entrenar el modelo lstm
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=False)
    # Hacer predicciones
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # Invertir las prediciones
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # Calcular la raíz cuadrada del error cuadratico medio
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Puntaje en entrenamiento: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Puntaje en prueba: %.2f RMSE' % (testScore))
    # correr las predicciones para plotear
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    #print(testPredict)
    # correr las prediciones para plotear
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
    #print('h', testPredictPlot)
    return [scaler.inverse_transform(dataset_copy), trainPredictPlot, testPredictPlot, [trainScore,testScore]]

def lstm_ventana(data_csv_np, epoch=200 ,porcentaje_entrenamiento=0.90, num_anteriores=3, num_capas=4):
    import numpy
    import pandas
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error


    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # Semilla para poder reproducir el experimento
    numpy.random.seed(7)
    # asignar dataset
    dataset = data_csv_np.astype('float32')
    dataset_copy = dataset.copy()
    # Dividir para train y split
    train_size = int(len(dataset) * porcentaje_entrenamiento) + 1 # agrego un nuevo valor en train
    test_size = len(dataset)+2 - train_size # Agrego uno en train y otro en test
    #-----------------------------------------------------------
    # Cuadrar los puntos para que se mostrar en la gráfica correctamente y no perder datos
    last_row = numpy.array(dataset.iloc[-1,:].values).reshape(1,1)
    last_train_row = numpy.array(dataset.iloc[train_size-1,:].values).reshape(1,1)
    dataset_split = dataset.iloc[:train_size-1,:]
    dataset_split = dataset_split.append(pandas.DataFrame(last_train_row))
    dataset_split2 = dataset.iloc[train_size-1:,:]
    dataset_split2 = dataset_split2.append(pandas.DataFrame(last_row))
    frames = [dataset_split, dataset_split2]
    dataset = pandas.concat(frames)
    #------------------------------------------------------------
    # Normalizar el dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset_copy = scaler.fit_transform(dataset_copy)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print('Valores en train: ',len(train)-1, 'Valores en test: ',len(test)-1)
    # reshape into X=t and Y=t+1
    look_back = num_anteriores
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(num_capas, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=False)
    # hacer predicciones
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invertir predicciones
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calcular la raiz cuadrada del error medio
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    #  correr las predicciones para plotear
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # correr las predicciones para plotear
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
    return [scaler.inverse_transform(dataset_copy), trainPredictPlot, testPredictPlot, [trainScore,testScore]]

def lstm_ventana_regresion(data_csv_np, epoch=200 ,porcentaje_entrenamiento=0.90, num_anteriores=3, num_capas=4):
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import pandas
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataset = data_csv_np.astype('float32')
    dataset_copy = dataset.copy()
    # Dividir para train y split
    train_size = int(len(dataset) * porcentaje_entrenamiento) + 1 # agrego un nuevo valor en train
    test_size = len(dataset)+2 - train_size # Agrego uno en train y otro en test
    #-----------------------------------------------------------
    # Cuadrar los puntos para que se mostrar en la gráfica correctamente y no perder datos
    last_row = numpy.array(dataset.iloc[-1,:].values).reshape(1,1)
    last_train_row = numpy.array(dataset.iloc[train_size-1,:].values).reshape(1,1)
    dataset_split = dataset.iloc[:train_size-1,:]
    dataset_split = dataset_split.append(pandas.DataFrame(last_train_row))
    dataset_split2 = dataset.iloc[train_size-1:,:]
    dataset_split2 = dataset_split2.append(pandas.DataFrame(last_row))
    frames = [dataset_split, dataset_split2]
    dataset = pandas.concat(frames)
    #------------------------------------------------------------
    # Normalizar el dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset_copy = scaler.fit_transform(dataset_copy)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print('Valores en train: ',len(train)-1, 'Valores en test: ',len(test)-1)
    # reshape into X=t and Y=t+1
    look_back = num_anteriores
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(num_capas, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=False)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
    return [scaler.inverse_transform(dataset_copy), trainPredictPlot, testPredictPlot, [trainScore,testScore]]

def lstm_memoria_lotes(data_csv_np, epoch=50 ,porcentaje_entrenamiento=0.80, num_anteriores=3, num_capas=4):
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import pandas
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataset = data_csv_np.astype('float32')
    dataset_copy = dataset.copy()
    # Dividir para train y split
    train_size = int(len(dataset) * porcentaje_entrenamiento) + 1 # agrego un nuevo valor en train
    test_size = len(dataset)+2 - train_size # Agrego uno en train y otro en test
    #-----------------------------------------------------------
    # Cuadrar los puntos para que se mostrar en la gráfica correctamente y no perder datos
    last_row = numpy.array(dataset.iloc[-1,:].values).reshape(1,1)
    last_train_row = numpy.array(dataset.iloc[train_size-1,:].values).reshape(1,1)
    dataset_split = dataset.iloc[:train_size-1,:]
    dataset_split = dataset_split.append(pandas.DataFrame(last_train_row))
    dataset_split2 = dataset.iloc[train_size-1:,:]
    dataset_split2 = dataset_split2.append(pandas.DataFrame(last_row))
    frames = [dataset_split, dataset_split2]
    dataset = pandas.concat(frames)
    #------------------------------------------------------------
    # Normalizar el dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset_copy = scaler.fit_transform(dataset_copy)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print('Valores en train: ',len(train)-1, 'Valores en test: ',len(test)-1)
    # reshape into X=t and Y=t+1
    look_back = num_anteriores
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(num_capas, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(100):
        model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size, verbose=False, shuffle=False)
        model.reset_states()
    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
    return [scaler.inverse_transform(dataset_copy), trainPredictPlot, testPredictPlot, [trainScore,testScore]]

def lstm_memoria_lotes_apilados(data_csv_np, epoch=5 ,porcentaje_entrenamiento=0.80, num_anteriores=3, num_capas=4):
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset

    dataset = data_csv_np.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * porcentaje_entrenamiento)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = num_anteriores
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(num_capas, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(num_capas, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(100):
        model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    return [scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot, [trainScore,testScore]]

def plot_localidades_crecimiento(estra,data_csv_localidad,tipo='dinamico'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    if tipo == 'dinamico':
        data_csv_localidad_copy = data_csv_localidad.copy()
        geo = estra[['Longitud','Latitud']].values
        geo = pd.DataFrame(geo, columns=['Longitud','Latitud'])
        IDs = estra.index.values
        Idtempd = pd.DataFrame(IDs, columns=['ID Localidad'])
        dft = pd.concat([Idtempd, geo], axis=1)
        result = pd.merge(data_csv_localidad_copy, dft, on='ID Localidad')
        for i,ID_ in enumerate(result.iloc[:,0]):
            result.iloc[i,0] = estra['Nombre Localidad'][estra.index==ID_].values[0]
        result.rename(columns={'Población Por localidad': 'Población anual', 'ID Localidad': 'Nombre localidad'}, inplace=True)
        data_csv_localidad_melt = result.melt(id_vars=['Nombre localidad', 
                                                       'AÑO', 
                                                       'Población anual',
                                                       'Longitud',
                                                       'Latitud'], var_name='MES', value_name='Residuos generados')
        return data_csv_localidad_melt
    
    elif tipo == 'estatico':
        for ano in [2012,2013,2014,2015,2016]:
            poblacion_data = data_csv_localidad[data_csv_localidad['AÑO']==ano].sort_values(by='ID Localidad')
            p = estra.plot(figsize=(12,12),subplots=True,kind="scatter", x="Longitud", y="Latitud",alpha=0.4,
                   s=poblacion_data["Población Por localidad"]/700,
                   label="Población Por localidad",
                   c=poblacion_data['Febrero'], 
                   cmap=plt.get_cmap("jet"), 
                   colorbar=True,
                   title= 'Cambio en población y desecho de residuos - '+str(ano))
            for t in range(20):
                plt.text(estra.Longitud.values[t],estra.Latitud.values[t],estra['Nombre Localidad'].values[t])

        plt.legend();
        plt.show();
        #plt.close()
    
def series(estra,data_csv,suavizado=1,tipo="zonas"):
    import numpy as np
    import pandas as pd
    #--------------
    if tipo == "zonas": 
        # Solo dejar valores de las series
        data_csv_copy = data_csv.copy()
        data_csv_copy = data_csv_copy.drop(columns = ['AÑO','Población Por localidad'])
        n = estra['Zona'].unique()
        zonas = np.sort(n[~pd.isna(n)])
        # Crear periodos de tiempo, longitud de las series
        years = np.sort(data_csv['AÑO'].unique())
        period = len(years)*12
        rng = pd.date_range('1/1/'+str(int(years[0])), periods=period, freq='M')
        # crear las series de datos y almacenarlas en un arreglo
        dict_series = {}
        series_list = []
        for ind, zona in enumerate(zonas):
            l = data_csv_copy[data_csv_copy.Zona == zona].values
            result = []
            for r in range(l.shape[0]):
                result = np.concatenate([result, np.asfarray(l[r,1:],float)])
            # Arreglar la series
            s = pd.Series(result, dtype=float, index=rng)
            # Suavizar la serie
            s = s.rolling(suavizado).mean()
            dict_series[zona] = s
            series_list.append([zona,s])
    elif tipo == "localidades":
        # Solo dejar valores de las series
        data_csv_copy = data_csv.copy()
        localidades_id = np.sort(data_csv_copy['ID Localidad'].unique())
        data_csv_copy = data_csv_copy.drop(columns = ['ID Localidad','AÑO','Población Por localidad'])
        # Crear periodos de tiempo, longitud de las series
        years = np.sort(data_csv['AÑO'].unique())
        period = len(years)*12
        rng = pd.date_range('1/1/'+str(int(years[0])), periods=period, freq='M')
        # crear las series de datos y almacenarlas en un arreglo
        dict_series = {}
        series_list = []
        for l_id in localidades_id:
            l = data_csv_copy[data_csv['ID Localidad'] == l_id].values
            result = []
            for r in range(l.shape[0]):
                result = np.concatenate([result, np.asfarray(l[r,:],float)])
            # Arreglar la series
            s = pd.Series(result, dtype=float, index=rng)
            # Suavizar la serie
            s = s.rolling(suavizado).mean()
            dict_series[estra['Nombre Localidad'][estra.index==l_id].values[0]] = s
            series_list.append([estra['Nombre Localidad'][estra.index==l_id].values[0],s])
    return dict_series, series_list

def decisiontrees(name, serie, porcentaje_entrenamiento=0.90, deep = [3,4,5,6]):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    import math
    
    rng = np.random.RandomState(1)
    lenght = serie.values.shape[0]
    n_train = int(porcentaje_entrenamiento * lenght)
    X = np.array(range(1,lenght+1)).reshape(lenght,1)
    y = serie.values.reshape(lenght,1)
    for d in deep:
        regressor = DecisionTreeRegressor(max_depth=d, random_state=0)
        regressor.fit(X[:n_train],y[:n_train])
        X_test_plot = np.arange(0.0, lenght, 0.01)[:, np.newaxis]
        y_predict_plot = regressor.predict(X_test_plot)
        y_predict = regressor.predict(X[n_train:])
        y_error = math.sqrt(mean_squared_error(y_predict , y[n_train:]))
        # Plot the results
        plt.figure(figsize=(12,6))
        plt.scatter(X,y, s=20, edgecolor="black",
            c="darkorange", label="datos")
        plt.plot(X_test_plot, y_predict_plot, color="cornflowerblue",
             label="max_depth="+str(d), linewidth=2)

        plt.xlabel("Meses")
        plt.ylabel("Residuos")
        plt.title(name+" - regresión usando Decision Tree - Error: "+str(y_error))
        plt.legend()
        plt.show();

def SVRegresion(name, serie, cv=25,n_entrenamientos=50, porcentaje_entrenamiento=0.90):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import math
    plt.figure(figsize=(12,6))
    rng = np.random.RandomState(1)
    lenght = serie.values.shape[0]
    n_train = int(porcentaje_entrenamiento * lenght)
    X = np.array(range(1,lenght+1)).reshape(lenght,1)
    y = serie.values.reshape(lenght,1).ravel()
    
    train_size = n_entrenamientos
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=cv,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e5],
                               "gamma": np.logspace(-4, 4, 25)})
    svr.fit(X[:n_train], y[:n_train])
    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    sv_score_train = svr.best_estimator_.score(X[:n_train], y[:n_train])
    sv_score_test = svr.best_estimator_.score(X[n_train:], y[n_train:])
    print("Score entrenamiento R^2: ", sv_score_train)
    print("Score pruebas R^2: ", sv_score_test)
    print("Support vector (Coeficiente): %.3f" % sv_ratio)
    X_test_plot = np.arange(0.0, lenght, 0.01)[:, np.newaxis]
    y_svr_plot = svr.predict(X_test_plot)
    X_test = X[n_train:]
    y_test_p = svr.predict(X_test)
    y_error_train = math.sqrt(mean_squared_error(svr.predict(X[:n_train]), y[:n_train]))
    y_error = math.sqrt(mean_squared_error(y[n_train:], y_test_p))
    plt.scatter(X, y, c='darkorange', label='datos', zorder=1,
                edgecolors=(0, 0, 0))
    plt.plot(X_test_plot, y_svr_plot, c='cornflowerblue',
         label='SVR')
    plt.plot(X_test, y_test_p, c='r',
         label='SVR test')
    plt.xlabel("Meses")
    plt.ylabel("Residuos")
    plt.title(name+" - regresión usando SVR - Error en train: "+str(y_error_train)+" - Error test: "+str(y_error))
    plt.legend()
    plt.show()
    
    
def iniciar():
    print('Inicio ....')
    tipo = input("Modelo a entrenar para zonas o localidades? = ")
    print("Tamaño de la ventana para suavizar las series entre 1 - 4 ")
    suavizado = input("Tamaño de la ventana para el suavizado de las series? = ")
    print("Modelos disponibles SVR y LSTM (con pasos de tiempo)")
    modelo_a_entrenar = input("Cuál modelo quiere entrenar? = ")
    print("Número de prediciones no superior a 24 meses")
    n_prediciones = input("Número de prediciones a realizar? (en meses) = ")
    if modelo_a_entrenar == 'SVR':
        print("Número de entrenamientos entre 100 - 250")
        n_entrenamientos = int(input("Número de entrenamientos? = "))
        print("Número de validaciones cruzadas entre 5 - 25 ")
        cv = int(input("Cuántas validaciones cruzadas? = "))
        print("OK... Parámetros guardados")
        return [tipo, suavizado, modelo_a_entrenar, n_entrenamientos, cv, n_prediciones]
    elif modelo_a_entrenar == 'LSTM':
        print("Número de capas entre 1-10")
        n_capas = int(input("Cuántas capas? = "))
        print("Número de entrenamiento entre 1 - 300")
        n_epoch = int(input("Cuántas epocas de entrenamiento? = "))
        print("Número de pasos anteriores entre 1 - 5")
        n_anteriores = int(input("Cuántos pasos anteriores? = "))
        print("OK... Parámetros guardados")
        return [tipo, suavizado, modelo_a_entrenar, n_capas, n_epoch, n_anteriores, n_prediciones]

def generar_series(estratificacion,data_csv_zona, data_csv_localidad, parametros):
    if parametros[0] == 'localidades':
        dict_, series_list = series(estratificacion,
                                               data_csv_localidad,
                                               suavizado=int(parametros[1]),
                                               tipo=parametros[0])
        return [dict_, series_list]
    if parametros[0] == 'zonas':
        dict_, series_list = series(estratificacion,
                                               data_csv_zona,
                                               suavizado=int(parametros[1]),
                                               tipo=parametros[0])
        return [dict_, series_list]

def SVRegresion_model(name, serie, cv=25,n_entrenamientos=50, porcentaje_entrenamiento=1.0, n_predicciones=12):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import math
    # figura
    plt.figure(figsize=(12,6))
    rng = np.random.RandomState(1)
    lenght = serie.values.shape[0]
    n_train = int(porcentaje_entrenamiento * lenght)
    X = np.array(range(1,lenght+1)).reshape(lenght,1)
    y = serie.values.reshape(lenght,1).ravel()
    
    train_size = n_entrenamientos
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=cv,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e5],
                               "gamma": np.logspace(-4, 4, 25)})
    svr.fit(X[:n_train], y[:n_train])
    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    sv_score_train = svr.best_estimator_.score(X[:n_train], y[:n_train])
    print("Score entrenamiento R^2: ", sv_score_train)
    print("Support vector (Coeficiente): %.3f" % sv_ratio)
    X_test_plot = np.arange(lenght+1, (lenght + n_predicciones)+1, 1)[:, np.newaxis]
    y_svr_plot = svr.predict(X_test_plot)
    y_error_train = math.sqrt(mean_squared_error(svr.predict(X[:n_train]), y[:n_train]))
    plt.scatter(X, y, c='darkorange', label='datos', zorder=1,
                edgecolors=(0, 0, 0))
    plt.plot(X_test_plot, y_svr_plot, c='r',
         label='predicciones')
    plt.plot(X, svr.predict(X[:n_train]), c='cornflowerblue',
         label='SVR train')
    plt.xlabel("Meses")
    plt.ylabel("Residuos")
    plt.title(name+" - regresión usando SVR - Error en train: "+str(y_error_train))
    plt.legend()
    plt.show()
    return y_svr_plot


def lstm_ventana_regresion_model(name, data_csv_np, epoch=200 ,porcentaje_entrenamiento=1.0, num_anteriores=3, num_capas=4, n_predicciones=12):
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import pandas
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataset = data_csv_np.astype('float32')
    dataset_copy = dataset.copy()
    # Dividir para train y split
    train_size = int(len(dataset) * porcentaje_entrenamiento)+1
    #-----------------------------------------------------------
    # Cuadrar los puntos para que se mostrar en la gráfica correctamente y no perder datos
    last_row = numpy.array(dataset.iloc[-1,:].values).reshape(1,1)
    dataset = dataset.append(pandas.DataFrame(last_row))
    #------------------------------------------------------------
    # Normalizar el dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset_copy = scaler.fit_transform(dataset_copy)
    train = dataset[0:train_size,:]
    test = dataset[-(num_anteriores+1):-1,:]
    print('Valores en train: ',len(train)-1)
    # reshape into X=t and Y=t+1
    look_back = num_anteriores
    trainX, trainY = create_dataset(train, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(test, (test.shape[1], test.shape[0], 1))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(num_capas, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=False)
    # make predictions
    trainPredict = model.predict(trainX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    predictions = []
    for p in range(n_predicciones):
        testPredict = model.predict(testX)
        predictions.append(testPredict)
        t = testX[0,:,0]
        t = list(numpy.delete(t,0,0))
        t.append(testPredict)
        testX[0,:,0] =  numpy.array(t)
    predicts = scaler.inverse_transform(numpy.array(predictions).flatten().reshape(len(predictions),1))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # plotting
    fig = plt.figure(figsize=(12,6))
    plt.title(name+ '- Error en train ' +str(trainScore))
    lenght = scaler.inverse_transform(dataset_copy).shape[0]
    X = numpy.array(range(1,lenght+1)).reshape(lenght,1)
    plt.scatter(X,scaler.inverse_transform(dataset_copy),c='darkorange', label='datos', zorder=1,
                edgecolors=(0, 0, 0));
    plt.plot(trainPredictPlot,c='cornflowerblue',
         label='LSTM');
    plt.plot(range(len(dataset)-1,len(dataset)-1+n_predicciones),predicts,c='r',
         label='Predicciones');
    plt.legend();plt.show();plt.close()
    return predicts
    
def entrenar_modelo(dic_series, series_list, parametros):
    import pandas as pd
    predicciones_lista = []
    if parametros[2] == 'SVR':
        keys = dic_series.keys()
        for k in keys:
            predicciones = SVRegresion_model(k, 
                                   dic_series[k].dropna(), 
                                   cv=parametros[4], 
                                   n_entrenamientos=parametros[3], 
                                   porcentaje_entrenamiento=1.0,
                                   n_predicciones=int(parametros[5]))
            predicciones_lista.append(predicciones)
        return predicciones_lista
    elif parametros[2] == 'LSTM':
        keys = list(dic_series.keys())
        for i,si in enumerate(series_list):
            series_list[i][1] = si[1].dropna()
        data_ = pd.DataFrame(series_list)
        for num_ in range(0,len(series_list)):
            data_i = pd.DataFrame(data_.iloc[num_,:].values[1].values)
            predicciones = lstm_ventana_regresion_model(keys[num_],
                                                               data_i, 
                                                               epoch=int(parametros[4]),
                                                               porcentaje_entrenamiento=1.0, 
                                                               num_anteriores=int(parametros[5]), 
                                                               num_capas=int(parametros[3]),
                                                               n_predicciones =int(parametros[6]))
            predicciones_lista.append(predicciones)
        return predicciones_lista
    
def estratificacion_residuos(tipo, estratificacion, caracterizacion, prediciones, data_csv_zona, data_csv_localidad):
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    
    residuos_estrato = defaultdict(list)
    if tipo == 'zonas':
        # poblaciones último año ordenadas por zona
        poblaciones = data_csv_zona[data_csv_zona['AÑO'] == 2016 ].sort_values(by=['Zona'])[data_csv_zona.keys()[-1]]
        predicciones_array = pd.DataFrame(np.array(prediciones))
    elif tipo == 'localidades':
        # poblaciones último año ordenadas por ID localidad
        poblaciones = data_csv_localidad[data_csv_localidad['AÑO'] == 2016 ].sort_values(by=['ID Localidad'])[data_csv_localidad.keys()[-1]]
        predicciones_array = pd.DataFrame(np.array(prediciones))
        poblaciones = data_csv_localidad[data_csv_localidad['AÑO'] == 2016 ].sort_values(by=['ID Localidad'])[data_csv_localidad.keys()[-1]]
        poblaciones_values = poblaciones.values.reshape(len(poblaciones),1)
        estratificaciones_pd = estratificacion.iloc[:,2:-3]
        #poblacion_estrato = pd.DataFrame(poblaciones_values*(estratificaciones_pd/100))
        for row in range(predicciones_array.shape[0]):
            for col in range(predicciones_array.shape[1]):
                ptoneladas = predicciones_array.iloc[row,col]
                l_t = []
                for l in range(estratificaciones_pd.shape[1]):
                    l_t.append(estratificaciones_pd.iloc[row,l]/100*ptoneladas)
                residuos_estrato[row+1].append(l_t)
        return residuos_estrato

def plot_residuos_estrato(estratificacion,residuos_estrato, parametros):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    if parametros[0] == 'zonas':
        pass
    elif parametros[0] == 'localidades':
        id_localidad = input("Cuál localidad (ID) quiere visualizar? = ")
        n_prediccion = input("Cuál predicción (mes) quiere visualizar? = ")
        nombre = estratificacion['Nombre Localidad'][estratificacion.index==int(id_localidad)].values[0]
        ax = pd.DataFrame(residuos_estrato[int(id_localidad)][int(n_prediccion)],columns=[nombre]).plot.bar();
        plt.rcParams["figure.figsize"] = [8,6];
        ax.set_ylabel('Residuos (Toneladas)');
        ax.set_xlabel('Estratos');
        ax.set_title('Residuos por estrato en la localidad '+ nombre);
        
def plot_caracterizacion(estratificacion, caracterizacion, residuos_estrato, parametros):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    if parametros[0] == 'zonas':
        pass
    elif parametros[0] == 'localidades':
        id_localidad = input("Cuál localidad (ID) quiere visualizar? = ")
        n_prediccion = input("Cuál predicción (mes) quiere visualizar? = ")
        n_estrato = input("Cuál es el estrato (1-6) que quiere visualizar? = ")
        nombre = estratificacion['Nombre Localidad'][estratificacion.index==int(id_localidad)].values[0]
        pr_estrato = residuos_estrato[int(id_localidad)][int(n_prediccion)][int(n_estrato)]
        ax = pd.DataFrame(pr_estrato*caracterizacion['Estrato '+n_estrato]/100).plot.bar();
        columns = list(caracterizacion.SUBCATEGORIA.values)
        ax.set_xticklabels(columns);
        plt.rcParams["figure.figsize"] = [12,8];
        ax.set_ylabel('Residuos (Toneladas)');
        ax.set_xlabel('Subcategorias');
        ax.set_title('Residuos por categorias en el estrato '+n_estrato+' de la localidad '+ nombre);
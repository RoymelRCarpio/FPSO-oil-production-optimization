# -*- coding: utf-8 -*-
"Otimização aplicando o Método classico de linearização por partes" 

import numpy as np
import pulp as lp
import timeit

"Exemplo simples para guia"
n_rows = 3 #Pontos de whp 
n_columns = 4 #Pontos de gl
qop = np.array([[100, 200, 300, 400],
                [ 50, 100, 150, 200],
                [ 25,  75, 100, 125]])

model_ex = lp.LpProblem("Example",lp.LpMaximize)

#Pesos
pesos = lp.LpVariable.dicts("Pesos",
                             ((i,j) for i in range(n_rows) for j in range(n_columns)),
                                lowBound=0,
                                upBound=1,
                                cat='Continuous')

#Segmentos nas linhas
y_row = lp.LpVariable.dicts("y_row",
                             ((i) for i in range(n_rows)),
                                lowBound=0,
                                upBound=1,
                                cat='Binary')

#Segmentos nas colunas
y_column = lp.LpVariable.dicts("y_column",
                             ((j) for j in range(n_columns-1)),
                                lowBound=0,
                                upBound=1,
                                cat='Binary')

# Objective Function
model_ex += (lp.lpSum([pesos[i,j]*qop[i,j] for i in range(n_rows) for j in range(n_columns)]))

# Restriçoes

for i in range(n_rows):  #Primeiro segmento de linearização
    model_ex += (pesos[i,0] <= y_row[i])  
    model_ex += (pesos[i,0] <= y_column[0])

for i in range(n_rows):  #Último segmento de linearização
    model_ex += (pesos[i,n_columns-1] <= y_row[i])  
    model_ex += (pesos[i,n_columns-1] <= y_column[n_columns-2])

for i in range(n_rows): #segmento intermedios de linearização
    for j in range(n_columns-2):
        model_ex += (pesos[i,j+1] <= y_row[i])  
        model_ex += (pesos[i,j+1] <= y_column[j]+y_column[j+1])

model_ex += (lp.lpSum([pesos[i,j] for i in range(n_rows) for j in range(n_columns)])==1)
model_ex += (lp.lpSum([y_row[i]] for i in range(n_rows))==1)
model_ex += (lp.lpSum([y_column[j]] for j in range(n_columns-1))==1)

# Limitando a funçao objetivo
model_ex += (lp.lpSum([pesos[i,j]*qop[i,j] for i in range(n_rows) for j in range(n_columns)])<=125)


model_ex.solve()
print ("Status Example:", lp.LpStatus[model_ex.status])
Pesos = [[pesos[i, j].varValue for i in range(n_rows)] for j in range(n_columns)]
Y_rows = [[y_row[i].varValue for i in range(n_rows)]]
Y_columns = [[y_column[j].varValue for j in range(n_columns-1)]]
print("Pesos:", Pesos)
print("Y_rows:", Y_rows)
print("Y_columns:", Y_columns)
print ("FO = ", lp.value(model_ex.objective))
print ("")
"Fim de Exemplo simples para guia"



"Començo do modelo do artigo"

''' Iniciando contabilizador de tempo '''
TempoInicio = timeit.default_timer()

''' Parametros do modelo'''

par_mod_well1 = np.array([[0,-0.0000825,-0.000000000038,26.443981756,0],
                        [0,0.0000221,-0.000000000202,24.642660534,0],
                        [0,0.00013070134351,-0.000000000415,22.800074621,0],
                        [0,0.00021222750339,-0.000000000553,21.180165758,0],
                        [0,0.00035470273205,-0.000000000785,18.042789067,0],
                        [0,0.00049069537006,-0.00000000103,15.045639362,0],
                        [0,0.00068056493238,-0.00000000137,11.526992533,0],
                        [0,0.0010654850544,-0.00000000211,6.1274822934,0]])
         
par_mod_well2 = np.array([[-5369.8246713,0.00094441433725,-0.00000000553,516.93820319,30000],
                        [-13108.167009,-0.0058137626012,0.00000000345,1221.6837799,60000],
                        [-10166.138816,-0.00084396809925,-0.0000000031,907.79082056,80000],
                        [-14808.472034,-0.0043398307907,0.00000000108,1323.1584005,100000],
                        [-15000,-0.0012403485474,-0.00000000352,1280.9935692,150000],
                        [-15000,-0.0009464945724,-0.00000000197,1247.3145346,210000],
                        [0,0,0,0,0],
                        [0,0,0,0,0]])
                        
par_mod_well3 = np.array([[322.06840985,0.000579,-0.00000000112,10.343279918,0],
                        [249.83830981,0.000646,-0.00000000126,14.107965073,0],
                        [125.97446158,0.000626,-0.00000000127,23.095255359,0],
                        [0,0.000594,-0.00000000126,32.408255051,0],
                        [0,0.000862,-0.00000000173,26.605146217,0],
                        [0,0.00108,-0.00000000209,21.2065865,0],
                        [0,0.00125,-0.00000000234,16.258950105,0],
                        [0,0.00178,-0.0000000034,8.3738765679,0]])                      

par_mod_well4 = np.array([[632.59624412,0.000833,-0.00000000191,10.51284994,0],
                        [540.03143753,0.00113,-0.00000000249,12.51025985,0],
                        [450.40971234,0.00139,-0.00000000298,14.50061337,0],
                        [340.36643956,0.00156,-0.0000000033,18.927147403,0],
                        [0,0.0016,-0.00000000337,40.776140529,0],
                        [0,0.00193,-0.00000000374,31.080506707,0],
                        [0,0.00211,-0.00000000387,22.930727398,0],
                        [0,0.00219,-0.00000000384,16.202373531,0]])  

par_mod_well5 = np.array([[232.65936607,0.00037984671382,-0.000000000887,8.3087994516,0],
                        [198.18800868,0.00045287514531,-0.00000000102,9.5789838195,0],
                        [149.11243862,0.00049483897412,-0.00000000111,12.366097848,0],
                        [0,0.00038541180556,-0.000000000969,25.03972338,0],
                        [0,0.00058495880845,-0.00000000132,21.24141143,0],
                        [0,0.00075759115872,-0.00000000161,17.631833035,0],
                        [0,0.00089569951131,-0.00000000182,14.277634234,0],
                        [0,0.00099359318863,-0.00000000194,11.249786062,0]])
                          
par_mod_well6 = np.array([[0,0.000891,-0.00000000169,12.01,0],
                        [0,0.00165,-0.00000000333,0.56,0],
                        [-1750.0594384,-0.00052,-0.000000000194,164.05465199,50000],
                        [-289.48183051,0.00102,-0.00000000219,22.39875594,50000],
                        [-1120.9031652,-0.000644,0.000000000415,104.50663995,150000],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0]])
                        
par_mod_well7 = np.array([[2454.4033146,0.000321,-0.00000000155,3.5598481307,0],
                        [2239.7233275,0.00115,-0.00000000291,6.951933884,0],
                        [1681.936899,0.00296,-0.0000000057,16.04511352,0],
                        [1034.3314385,0.00452,-0.00000000802,30.764019261,0],
                        [108.27456556,0.00513,-0.00000000875,76.908137209,0],
                        [0,0.00628,-0.0000000103,50.047502676,0],
                        [0,0.00683,-0.0000000108,21.683679853,0],
                        [0,0.00627,-0.0000000092,6.3071767584,0]])
                        
par_mod_well8 = np.array([[724.41852012,0.0014654641231,-0.00000000252,8.440529658,0],
                        [611.81183223,0.001659298368,-0.00000000277,10.745122638,0],
                        [501.70916143,0.0018003169936,-0.00000000297,13.418820666,0],
                        [0,0.0013639401363,-0.0000000023,54.775378272,0],
                        [0,0.0018855640965,-0.00000000312,39.864343139,0],
                        [0,0.0023439279091,-0.00000000373,21.019296466,0],
                        [-1473.1063015,0.00031343112959,-0.000000000448,155.15997814,50000],
                        [-2226.4704894,-0.00036580263817,0.00000000043,216.74008484,100000]])                                                                        

WHP = np.array([5,10,15,20,30,40,50,60]) 

n_whp = len(WHP)

GOR = np.array([89.9, 100.0, 112.1, 115.9, 118.5, 145.1, 105.0, 128.5])              

WC = np.array([0.525, 0.019, 0.540, 0.448, 0.665, 0.021, 0.141, 0.361])

''' Fim de Parametros do modelo'''

''' Gerando os pontos da linearização'''
n_wells = 8 # Número de poços
npl = 10 #número de pontos da linearização #Marcador
qgl = np.linspace(0,350000,npl)
# Allocando as matrices
qo_well1 = np.zeros([n_whp,npl])
qo_well2 = np.zeros([n_whp,npl])
qo_well3 = np.zeros([n_whp,npl])
qo_well4 = np.zeros([n_whp,npl])
qo_well5 = np.zeros([n_whp,npl])
qo_well6 = np.zeros([n_whp,npl])
qo_well7 = np.zeros([n_whp,npl])
qo_well8 = np.zeros([n_whp,npl])
# Preenchendo as matrizes de qo(whp,gl)
for i in range(n_whp):
    for j in range(npl):
        qo_well1[i,j] = par_mod_well1[i,0]+par_mod_well1[i,1]*qgl[j]+par_mod_well1[i,2]*qgl[j]**2+par_mod_well1[i,3]*np.log(qgl[j]+1)
        if qgl[j] < par_mod_well2[i,4]:
            qo_well2[i,j] = 0
        else:
            qo_well2[i,j] = par_mod_well2[i,0]+par_mod_well2[i,1]*qgl[j]+par_mod_well2[i,2]*qgl[j]**2+par_mod_well2[i,3]*np.log(qgl[j]+1)
        qo_well3[i,j] = par_mod_well3[i,0]+par_mod_well3[i,1]*qgl[j]+par_mod_well3[i,2]*qgl[j]**2+par_mod_well3[i,3]*np.log(qgl[j]+1)
        qo_well4[i,j] = par_mod_well4[i,0]+par_mod_well4[i,1]*qgl[j]+par_mod_well4[i,2]*qgl[j]**2+par_mod_well4[i,3]*np.log(qgl[j]+1)
        qo_well5[i,j] = par_mod_well5[i,0]+par_mod_well5[i,1]*qgl[j]+par_mod_well5[i,2]*qgl[j]**2+par_mod_well5[i,3]*np.log(qgl[j]+1)
        if qgl[j] < par_mod_well6[i,4]:
            qo_well6[i,j] = 0
        else:
            qo_well6[i,j] = par_mod_well6[i,0]+par_mod_well6[i,1]*qgl[j]+par_mod_well6[i,2]*qgl[j]**2+par_mod_well6[i,3]*np.log(qgl[j]+1)
        qo_well7[i,j] = par_mod_well7[i,0]+par_mod_well7[i,1]*qgl[j]+par_mod_well7[i,2]*qgl[j]**2+par_mod_well7[i,3]*np.log(qgl[j]+1)
        if qgl[j] < par_mod_well8[i,4]:
            qo_well8[i,j] = 0
        else:
            qo_well8[i,j] = par_mod_well8[i,0]+par_mod_well8[i,1]*qgl[j]+par_mod_well8[i,2]*qgl[j]**2+par_mod_well8[i,3]*np.log(qgl[j]+1)
        
# Gráfico
#plt.plot(qgl, np.transpose(qo_well8))
#plt.title("Well 8")
#plt.xlabel("Gas injection")
#plt.ylabel("Oil production")
#plt.show()
'''Fim de Gerando os pontos da linearização'''
 
''' Modelo e otimização da plataforma em PuLP'''

# Creando um problema de maximização
model = lp.LpProblem("Artigo",lp.LpMaximize)

'''Variaveis do modelo'''
# Pessos da linearização (matriz[[n_whp,npl]])
#Pesos
landa_well1 = lp.LpVariable.dicts("landa_well1",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well2 = lp.LpVariable.dicts("landa_well2",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well3 = lp.LpVariable.dicts("landa_well3",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well4 = lp.LpVariable.dicts("landa_well4",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well5 = lp.LpVariable.dicts("landa_well5",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well6 = lp.LpVariable.dicts("landa_well6",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well7 = lp.LpVariable.dicts("landa_well7",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')
landa_well8 = lp.LpVariable.dicts("landa_well8",((i,j) for i in range(n_whp) for j in range(npl)),lowBound=0,upBound=1,cat='Continuous')

# Variaveis enteiras que indican os valores de whp (vector [n_whp])
y_whp_well1 = lp.LpVariable.dicts("y_whp_well1",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well2 = lp.LpVariable.dicts("y_whp_well2",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well3 = lp.LpVariable.dicts("y_whp_well3",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well4 = lp.LpVariable.dicts("y_whp_well4",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well5 = lp.LpVariable.dicts("y_whp_well5",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well6 = lp.LpVariable.dicts("y_whp_well6",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well7 = lp.LpVariable.dicts("y_whp_well7",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')
y_whp_well8 = lp.LpVariable.dicts("y_whp_well8",((i) for i in range(n_whp)),lowBound=0,upBound=1,cat='Binary')

# Variaveis enteiras que indican os segmentos de gl (vector [npl-1])
y_gl_well1 = lp.LpVariable.dicts("y_gl_well1",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well2 = lp.LpVariable.dicts("y_gl_well2",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well3 = lp.LpVariable.dicts("y_gl_well3",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well4 = lp.LpVariable.dicts("y_gl_well4",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well5 = lp.LpVariable.dicts("y_gl_well5",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well6 = lp.LpVariable.dicts("y_gl_well6",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well7 = lp.LpVariable.dicts("y_gl_well7",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')
y_gl_well8 = lp.LpVariable.dicts("y_gl_well8",((j) for j in range(npl-1)),lowBound=0,upBound=1,cat='Binary')

# Variaveis inteiras que indicam o estado del poço (aberto ou fechado)
#estado_well1 = m.Var(value =1, lb=0, ub=1, integer=True)
#estado_well2 = m.Var(value =1, lb=0, ub=1, integer=True) 
#estado_well3 = m.Var(value =1, lb=0, ub=1, integer=True) 
#estado_well4 = m.Var(value =1, lb=0, ub=1, integer=True)
#estado_well5 = m.Var(value =1, lb=0, ub=1, integer=True)
#estado_well6 = m.Var(value =1, lb=0, ub=1, integer=True)
#estado_well7 = m.Var(value =1, lb=0, ub=1, integer=True)
#estado_well8 = m.Var(value =1, lb=0, ub=1, integer=True)  

#Vazao de gas lift (vector [n_wells])
Qgl = lp.LpVariable.dicts("Qgl",((l) for l in range(n_wells)),lowBound=0)

#pressões na cabeça dos poços (vector [n_wells])
whp = lp.LpVariable.dicts("whp",((l) for l in range(n_wells)),lowBound=0)

#Vazao de oleo (vector [n_wells])
qo = lp.LpVariable.dicts("qo",((l) for l in range(n_wells)),lowBound=0)

#Vazões de gas e agua produzidas em cada poço (vector [n_wells])
qg = lp.LpVariable.dicts("qg",((l) for l in range(n_wells)),lowBound=0)
qw = lp.LpVariable.dicts("qw",((l) for l in range(n_wells)),lowBound=0)

# Vazões totais de oleo, agua, liquido, gas e gas lift
qo_total = lp.LpVariable("qo_total",lowBound=0)
qw_total = lp.LpVariable("qw_total",lowBound=0)
ql_total = lp.LpVariable("ql_total",lowBound=0)
qg_total = lp.LpVariable("qg_total",lowBound=0)
qgl_total = lp.LpVariable("qgl_total",lowBound=0)

# Vazao de gas tratado
qg_tratado = lp.LpVariable("qg_tratado",lowBound=0)

# Vazao de gas exportado
qg_exp = lp.LpVariable("qg_exp",lowBound=0)

# Restricoes da plataforma
qg_turbina = lp.LpVariable("qg_turbina",lowBound=1e5,upBound=1e5)
qg_flare = lp.LpVariable("qg_flare",lowBound=0,upBound=1e4)
'''Fim de Variaveis do modelo'''

'''Equaçoes do modelo'''
### Restriçoes que devem satisfazer os pesos da linearização ###

# Um valor de whp deve estar ativo
model += (lp.lpSum([y_whp_well1[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well2[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well3[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well4[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well5[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well6[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well7[i]] for i in range(n_whp))==1)
model += (lp.lpSum([y_whp_well8[i]] for i in range(n_whp))==1)

# Um segmento da linearização de gas lif deve estar ativo
model += (lp.lpSum([y_gl_well1[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well2[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well3[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well4[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well5[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well6[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well7[j]] for j in range(npl-1))==1)
model += (lp.lpSum([y_gl_well8[j]] for j in range(npl-1))==1)

# A soma dos pesos deve ser 1, ou zero em caso do que o poço esteja fechado
model += (lp.lpSum([landa_well1[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well1)
model += (lp.lpSum([landa_well2[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well2)
model += (lp.lpSum([landa_well3[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well3)
model += (lp.lpSum([landa_well4[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well4)
model += (lp.lpSum([landa_well5[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well5)
model += (lp.lpSum([landa_well6[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well6)
model += (lp.lpSum([landa_well7[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well7)
model += (lp.lpSum([landa_well8[i,j] for i in range(n_whp) for j in range(npl)])==1) #estado_well8)

# Primeiro peso somente pode ser diferente de zero se o primero segmento de gl estiver activo 
for i in range(n_whp):
    model += (landa_well1[i,0] <= y_gl_well1[0])
    model += (landa_well2[i,0] <= y_gl_well2[0])
    model += (landa_well3[i,0] <= y_gl_well3[0])
    model += (landa_well4[i,0] <= y_gl_well4[0])
    model += (landa_well5[i,0] <= y_gl_well5[0])
    model += (landa_well6[i,0] <= y_gl_well6[0])
    model += (landa_well7[i,0] <= y_gl_well7[0])
    model += (landa_well8[i,0] <= y_gl_well8[0])

# Pesos intermedios somente podem ser diferentes de zero se o segmento anterior ou o segmento correpondente estão ativos
for i in range(n_whp):
    for j in range(npl-2):
        model += (landa_well1[i,j+1] <= y_gl_well1[j]+y_gl_well1[j+1])
        model += (landa_well2[i,j+1] <= y_gl_well2[j]+y_gl_well2[j+1])
        model += (landa_well3[i,j+1] <= y_gl_well3[j]+y_gl_well3[j+1])
        model += (landa_well4[i,j+1] <= y_gl_well4[j]+y_gl_well4[j+1])
        model += (landa_well5[i,j+1] <= y_gl_well5[j]+y_gl_well5[j+1])
        model += (landa_well6[i,j+1] <= y_gl_well6[j]+y_gl_well6[j+1])
        model += (landa_well7[i,j+1] <= y_gl_well7[j]+y_gl_well7[j+1])
        model += (landa_well8[i,j+1] <= y_gl_well8[j]+y_gl_well8[j+1])

# Último peso somente pode ser diferente de zero se o ultimo segmento de gl estiver ativo
for i in range(n_whp):
    model += (landa_well1[i,npl-1] <= y_gl_well1[npl-2])
    model += (landa_well2[i,npl-1] <= y_gl_well2[npl-2])
    model += (landa_well3[i,npl-1] <= y_gl_well3[npl-2])
    model += (landa_well4[i,npl-1] <= y_gl_well4[npl-2])
    model += (landa_well5[i,npl-1] <= y_gl_well5[npl-2])
    model += (landa_well6[i,npl-1] <= y_gl_well6[npl-2])
    model += (landa_well7[i,npl-1] <= y_gl_well7[npl-2])
    model += (landa_well8[i,npl-1] <= y_gl_well8[npl-2])

# Somente pode ser diferente de zero o peso que corresponde com um whp activo
for i in range(n_whp):
    for j in range(npl):
        model += (landa_well1[i,j] <= y_whp_well1[i])
        model += (landa_well2[i,j] <= y_whp_well2[i])
        model += (landa_well3[i,j] <= y_whp_well3[i])
        model += (landa_well4[i,j] <= y_whp_well4[i])
        model += (landa_well5[i,j] <= y_whp_well5[i])
        model += (landa_well6[i,j] <= y_whp_well6[i])
        model += (landa_well7[i,j] <= y_whp_well7[i])
        model += (landa_well8[i,j] <= y_whp_well8[i])
### Fin de Restriçoes que devem satisfazer os pesos da linearização ###

# Vazão de gas lift
model += (Qgl[0] == lp.lpSum([landa_well1[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[1] == lp.lpSum([landa_well2[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[2] == lp.lpSum([landa_well3[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[3] == lp.lpSum([landa_well4[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[4] == lp.lpSum([landa_well5[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[5] == lp.lpSum([landa_well6[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[6] == lp.lpSum([landa_well7[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))
model += (Qgl[7] == lp.lpSum([landa_well8[i,j]*qgl[j] for i in range(n_whp) for j in range(npl)]))

# Pressão na cabeça do poço
model += (whp[0] == lp.lpSum([landa_well1[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[1] == lp.lpSum([landa_well2[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[2] == lp.lpSum([landa_well3[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[3] == lp.lpSum([landa_well4[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[4] == lp.lpSum([landa_well5[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[5] == lp.lpSum([landa_well6[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[6] == lp.lpSum([landa_well7[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))
model += (whp[7] == lp.lpSum([landa_well8[i,j]*WHP[i] for i in range(n_whp) for j in range(npl)]))

# Vazao de oleo
model += (qo[0] == lp.lpSum([landa_well1[i,j]*qo_well1[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[1] == lp.lpSum([landa_well2[i,j]*qo_well2[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[2] == lp.lpSum([landa_well3[i,j]*qo_well3[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[3] == lp.lpSum([landa_well4[i,j]*qo_well4[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[4] == lp.lpSum([landa_well5[i,j]*qo_well5[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[5] == lp.lpSum([landa_well6[i,j]*qo_well6[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[6] == lp.lpSum([landa_well7[i,j]*qo_well7[i,j] for i in range(n_whp) for j in range(npl)]))
model += (qo[7] == lp.lpSum([landa_well8[i,j]*qo_well8[i,j] for i in range(n_whp) for j in range(npl)]))

# Vazao de gas e agua produzidas
for l in range(n_wells):
    model += (qg[l] == qo[l]*GOR[l])
    model += (qw[l] == WC[l]/(1-WC[l])*qo[l])

# Vazões totais de oleo, agua, liquido, gas e gas lift
model += (qo_total == lp.lpSum([qo[l] for l in range(n_wells)]))
model += (qw_total == lp.lpSum([qw[l] for l in range(n_wells)]))
model += (ql_total == qo_total + qw_total)
model += (qg_total == lp.lpSum([qg[l] for l in range(n_wells)]))
model += (qgl_total == lp.lpSum([Qgl[l] for l in range(n_wells)]))

# Vazao de gas tratado
model += (qg_tratado == qg_total + qgl_total - qg_flare)

# Vazao de gas exportado
model += (qg_exp == qg_total - qg_flare - qg_turbina)

'''Restriçoes da plataforma'''
model += (qw_total <= 5000) #5000
model += (ql_total <= 20000) #20000
model += (qg_tratado <= 3000000) #3000000
model += (qg_flare <= 10000) #10000
model += (qg_exp <=610000) #1000000
'''Fim de Restrições da plataforma'''

'''Restrição segunda etapa otimização'''
model += (qo_total>=6544.52835-5)  #Marcador

'''Funçao objetivo'''
#model += (qo_total)
model += -(qgl_total+qg_flare)

'''Resolvendo o problema'''
model.solve()
print("Status Plataforma:", lp.LpStatus[model.status])
   

"""                

'''Evaluando o resultado linear no modelo não linear'''
qo_well1_real = m.Intermediate(par_mod_well1[0,0]+par_mod_well1[0,1]*(qgl_well1)+par_mod_well1[0,2]*(qgl_well1)**2+par_mod_well1[0,3]*m.log((qgl_well1)+1))
#qo_well2_real = m.Intermediate(par_mod_well2[0,0]+par_mod_well2[0,1]*(qgl_well2)+par_mod_well2[0,2]*(qgl_well2)**2+par_mod_well2[0,3]*m.log((qgl_well2)+1))
qo_well2_real = m.Intermediate(0.5*(1+m.tanh(qgl_well2-par_mod_well2[0,4]))*(par_mod_well2[0,0]+par_mod_well2[0,1]*qgl_well2+par_mod_well2[0,2]*qgl_well2**2+par_mod_well2[0,3]*m.log(qgl_well2+1)))
qo_well3_real = m.Intermediate(par_mod_well3[0,0]+par_mod_well3[0,1]*(qgl_well3)+par_mod_well3[0,2]*(qgl_well3)**2+par_mod_well3[0,3]*m.log((qgl_well3)+1))
qo_well4_real = m.Intermediate(par_mod_well4[0,0]+par_mod_well4[0,1]*(qgl_well4)+par_mod_well4[0,2]*(qgl_well4)**2+par_mod_well4[0,3]*m.log((qgl_well4)+1))
qo_well5_real = m.Intermediate(par_mod_well5[2,0]+par_mod_well5[2,1]*(qgl_well5)+par_mod_well5[2,2]*(qgl_well5)**2+par_mod_well5[2,3]*m.log((qgl_well5)+1))
#qo_well6_real = m.Intermediate(par_mod_well6[0,0]+par_mod_well6[0,1]*(qgl_well6)+par_mod_well6[0,2]*(qgl_well6)**2+par_mod_well6[0,3]*m.log((qgl_well6)+1))
qo_well6_real = m.Intermediate(0.5*(1+m.tanh(qgl_well6-par_mod_well6[0,4]))*(par_mod_well6[0,0]+par_mod_well6[0,1]*qgl_well6+par_mod_well6[0,2]*qgl_well6**2+par_mod_well6[0,3]*m.log(qgl_well6+1)))
qo_well7_real = m.Intermediate(par_mod_well7[0,0]+par_mod_well7[0,1]*(qgl_well7)+par_mod_well7[0,2]*(qgl_well7)**2+par_mod_well7[0,3]*m.log((qgl_well7)+1))
#qo_well8_real = m.Intermediate(par_mod_well8[0,0]+par_mod_well8[0,1]*(qgl_well8)+par_mod_well8[0,2]*(qgl_well8)**2+par_mod_well8[0,3]*m.log((qgl_well8)+1))
qo_well8_real = m.Intermediate(0.5*(1+m.tanh(qgl_well8-par_mod_well8[0,4]))*(par_mod_well8[0,0]+par_mod_well8[0,1]*qgl_well8+par_mod_well8[0,2]*qgl_well8**2+par_mod_well8[0,3]*m.log(qgl_well8+1)))
qo_total_real = m.Intermediate(qo_well1_real+qo_well2_real+qo_well3_real+qo_well4_real+qo_well5_real+qo_well6_real+qo_well7_real+qo_well8_real)
qw_total_real = m.Intermediate(WC[0]/(1-WC[0])*qo_well1_real+WC[1]/(1-WC[1])*qo_well2_real+WC[2]/(1-WC[2])*qo_well3_real+WC[3]/(1-WC[3])*qo_well4_real +
                               WC[4]/(1-WC[4])*qo_well5_real+WC[5]/(1-WC[5])*qo_well6_real+WC[6]/(1-WC[6])*qo_well7_real+WC[7]/(1-WC[7])*qo_well8_real)
qg_total_real = m.Intermediate(GOR[0]*qo_well1_real+GOR[1]*qo_well2_real+GOR[2]*qo_well3_real+GOR[3]*qo_well4_real+
                               GOR[4]*qo_well5_real+GOR[5]*qo_well6_real+GOR[6]*qo_well7_real+GOR[7]*qo_well8_real)
qg_tratado_real = m.Intermediate(qg_total_real + qgl_total- qg_flare)
qg_exp_real =  m.Intermediate(qg_total_real - qg_flare - qg_turbina)

#Solve simulation
m.solve(disp=False)
"""
#Results
print('')
print('Results')

print('qgl_well1: ' + str(Qgl[0].varValue))
print('qgl_well2: ' + str(Qgl[1].varValue))
print('qgl_well3: ' + str(Qgl[2].varValue))
print('qgl_well4: ' + str(Qgl[3].varValue))
print('qgl_well5: ' + str(Qgl[4].varValue))
print('qgl_well6: ' + str(Qgl[5].varValue))
print('qgl_well7: ' + str(Qgl[6].varValue))
print('qgl_well8: ' + str(Qgl[7].varValue))

print('whp_well1: ' + str(whp[0].varValue))
print('whp_well2: ' + str(whp[1].varValue))
print('whp_well3: ' + str(whp[2].varValue))
print('whp_well4: ' + str(whp[3].varValue))
print('whp_well5: ' + str(whp[4].varValue))
print('whp_well6: ' + str(whp[5].varValue))
print('whp_well7: ' + str(whp[6].varValue))
print('whp_well8: ' + str(whp[7].varValue))

print('qo_well1: ' + str(qo[0].varValue))
print('qo_well2: ' + str(qo[1].varValue))
print('qo_well3: ' + str(qo[2].varValue))
print('qo_well4: ' + str(qo[3].varValue))
print('qo_well5: ' + str(qo[4].varValue))
print('qo_well6: ' + str(qo[5].varValue))
print('qo_well7: ' + str(qo[6].varValue))
print('qo_well8: ' + str(qo[7].varValue))

print('qo_total: ' + str(qo_total.varValue))
print('qw_total: ' + str(qw_total.varValue))
print('ql_total: ' + str(ql_total.varValue))
print('qg_total: ' + str(qg_total.varValue))
print('qgl_total: ' + str(qgl_total.varValue))

print('qg_flare: ' + str(qg_flare.varValue))
print('qg_tratado: ' + str(qg_tratado.varValue))
print('qg_exp: ' + str(qg_exp.varValue))
print('FO: ' + str(lp.value(model.objective)))

''' Finalizando contabilizador de tempo '''
TempoFim = timeit.default_timer()
print('Tempo de execução: %f' % (TempoFim - TempoInicio))

'''
print("")

print('qo_well1_real: ' + str(qo_well1_real.value))
print('qo_well2_real: ' + str(qo_well2_real.value))
print('qo_well3_real: ' + str(qo_well3_real.value))
print('qo_well4_real: ' + str(qo_well4_real.value))
print('qo_well5_real: ' + str(qo_well5_real.value))
print('qo_well6_real: ' + str(qo_well6_real.value))
print('qo_well7_real: ' + str(qo_well7_real.value))
print('qo_well8_real: ' + str(qo_well8_real.value))
print('qo_total_real: ' + str(qo_total_real.value))
print('qw_total_real: ' + str(qw_total_real.value))
print('qg_tratado_real: ' + str(qg_tratado_real.value))
print('qg_exp_real: ' + str(qg_exp_real.value))
'''
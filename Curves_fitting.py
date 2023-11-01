# -*- coding: utf-8 -*-
# Interpolador Mutilinear utilizando um arquivo de dados externo 

import os
import numpy as np
import timeit
from gekko import GEKKO   
import matplotlib.pyplot as plt

''' Iniciando contabilizador de tempo '''
TempoInicio = timeit.default_timer()

''' Indicando a pasta onde estão as tabelas '''
dir = os.getcwd() # cwd = current working directory
Ndir = dir+'\Tabelas'
os.chdir(Ndir)
''' Fim de Indicando a pasta onde estão as tabelas '''

''' Carregando os dados dos arquivos externos para variaveis '''
# Carregando  os dados do arquivo oilWells
dados = open("oilWells.tab").readlines() # Abrir arquivo "oliwells"
n_wells = len(dados)-2  # Número de poços (eliminando as duas primeiras linhas)
gor = np.zeros([n_wells]) #Alocando a variavel gor
wc = np.zeros([n_wells]) #Alocando a variavel wc
for i in range(n_wells):
    linha = dados[i+2].split() #Para separar os dados
    gor[i] = float(linha[1]) #Variavel gor
    wc[i] = float(linha[2]) #Variavel wc
# Função para carregar os dados dos arquivos das curvas de eficiencia
def curva_eff(nome_arquivo):
    dados = open(nome_arquivo).readlines() # Abrir arquivo
    n_pontos = len(dados)-2  # Número de pontos (eliminando as duas primeiras linhas)
    matriz = np.zeros([n_pontos,3]) #Alocando a matriz para armazenar gl, whp, qo
    for i in range(n_pontos):
        linha = dados[i+2].split() #Para separar os dados
        matriz[i,0] = float(linha[0]) #Variavel gl
        matriz[i,1] = float(linha[1]) #Variavel whp
        matriz[i,2] = float(linha[2]) #Variavel qo
    return matriz
# Carregando os dados dos arquivos das curvas de eficiencia
well1 = curva_eff("Well1.tab")
well2 = curva_eff("Well2.tab")
well3 = curva_eff("Well3.tab")
well4 = curva_eff("Well4.tab")
well5 = curva_eff("Well5.tab")
well6 = curva_eff("Well6.tab")
well7 = curva_eff("Well7.tab")
well8 = curva_eff("Well8.tab")
''' Fin de Carregando os dados dos arquivos externos para variaveis '''

''' Função para determiar o numero de pontos de gas lift de um poço'''
def f_np_gl(well):
    np_gl = 1
    for i in range(len(well)-1):
        if well[i,0]!=well[i+1,0]:
            np_gl = np_gl + 1
    return np_gl
''' Fim de Função para determiar o numero de pontos de gas lift de um poço'''

''' Função para determiar o numero de pontos de whp de um poço'''
def f_np_whp(well):
    np_whp = 1
    i=0
    while well[i+1,1] != well[0,1]:
        np_whp = np_whp +1
        i = i + 1
    return np_whp
''' Fim de Função para determiar o numero de pontos de whp de um poço'''

''' Função para determinar o vetor com os valores de gl de cada poço'''
def f_glp_well(well,np_gl):
    glp_well = np.zeros(np_gl)
    j = 0
    glp_well[j] = well[0,0]
    for i in range(len(well)-1):
        if well[i,0]!=well[i+1,0]:
            j = j + 1
            glp_well[j] = well[i+1,0]
    return glp_well
''' Fim de Função para determinar o vetor com os valores de gl de cada poço'''
 
''' Funçao para determinar os valores de qo para cada ponto de gas lift e whp qo(np_whp,np_gl)'''
def f_qo_well(well, np_gl, np_whp):
    qgl_well = np.zeros([np_whp,np_gl])
    for i in range(np_gl):
        qgl_well[:,i] = well[i*(np_whp):i*(np_whp)+np_whp,2]
    return qgl_well
''' Fim de Funçao para determinar os valores de qo para cada ponto de gas lift e whp qo(np_whp,np_gl)'''
 
''' Parametros do modelo'''
# Numero de pontos de gas lift
np_gl_well1 = f_np_gl(well1)
np_gl_well2 = f_np_gl(well2)
np_gl_well3 = f_np_gl(well3)
np_gl_well4 = f_np_gl(well4)
np_gl_well5 = f_np_gl(well5)
np_gl_well6 = f_np_gl(well6) 
np_gl_well7 = f_np_gl(well7)
np_gl_well8 = f_np_gl(well8)

# Numero de pontos de whp
np_whp_well1 = f_np_whp(well1)
np_whp_well2 = f_np_whp(well2)
np_whp_well3 = f_np_whp(well3)
np_whp_well4 = f_np_whp(well4)
np_whp_well5 = f_np_whp(well5)
np_whp_well6 = f_np_whp(well6)
np_whp_well7 = f_np_whp(well7)
np_whp_well8 = f_np_whp(well8)

# Valores de gas lift
glp_well1 = f_glp_well(well1,np_gl_well1)
glp_well2 = f_glp_well(well2,np_gl_well2)
glp_well3 = f_glp_well(well3,np_gl_well3)
glp_well4 = f_glp_well(well4,np_gl_well4)
glp_well5 = f_glp_well(well5,np_gl_well5)
glp_well6 = f_glp_well(well6,np_gl_well6)
glp_well7 = f_glp_well(well7,np_gl_well7)
glp_well8 = f_glp_well(well8,np_gl_well8)

# Valores de whp
whpp_well1 = well1[0:np_whp_well1,1]
whpp_well2 = well2[0:np_whp_well2,1]
whpp_well3 = well3[0:np_whp_well3,1]
whpp_well4 = well4[0:np_whp_well4,1]
whpp_well5 = well5[0:np_whp_well5,1]
whpp_well6 = well6[0:np_whp_well6,1]
whpp_well7 = well7[0:np_whp_well7,1]
whpp_well8 = well8[0:np_whp_well8,1]

# Matriz de valores de qo qo(np_whp,np_gl)
qop_well1 = f_qo_well(well1, np_gl_well1, np_whp_well1)
qop_well2 = f_qo_well(well2, np_gl_well2, np_whp_well2)
qop_well3 = f_qo_well(well3, np_gl_well3, np_whp_well3)
qop_well4 = f_qo_well(well4, np_gl_well4, np_whp_well4)
qop_well5 = f_qo_well(well5, np_gl_well5, np_whp_well5)
qop_well6 = f_qo_well(well6, np_gl_well6, np_whp_well6)
qop_well7 = f_qo_well(well7, np_gl_well7, np_whp_well7)
qop_well8 = f_qo_well(well8, np_gl_well8, np_whp_well8)
''' Fim de Parametros do modelo'''

'''Estimação dos parametros do modelo'''
#Initialize Model
m = GEKKO(remote=False)

#Variables
a = m.Var(value=0,lb=-50000,ub=10000)
b = m.Var(value=0,lb=-10,ub=10)
c = m.Var(value=0,lb=-10,ub=10)
d = m.Var(value=0,lb=-1000,ub=9500)
qo_model = m.Array(m.Var,np_gl_well2-6)
MSE_vector = m.Array(m.Var,np_gl_well2-6)
MSE = m.Var(value=0,lb=0)

#Equations
for i in range(np_gl_well2-6):
    m.Equation(qo_model[i] == a + b*glp_well2[i+6] + c*glp_well2[i+6]**2 + d*np.log(glp_well2[i+6]+1)) #Modelo poço surgente
    m.Equation(MSE_vector[i] == (qo_model[i] - qop_well2[11,i+6])**2)

m.Equation(MSE == m.sum(MSE_vector))

#Objective
m.Obj(MSE)
    
#Set global options
m.options.IMODE = 3 #steady state optimization = 3, simulation = 1

#Solve simulation
m.solve()

#Results
print('')
print('Results')
print('a: ' + str(a.value))
print('b: ' + str(b.value))
print('c: ' + str(c.value))
print('d: ' + str(d.value))
print('MSE: ' + str(MSE.value))

#Resposta do modelo
qo_fitted = np.zeros(100)
gl = np.linspace(0,350000,100)
#Binary = 0.5 + 0.5 * np.sign(0-gl)
#qo_fitted = Binary*0 + (1-Binary)*(a.value + b.value*gl + c.value*gl**2 + d.value*np.log(gl+1))
qo_fitted = 0.5*(1+np.tanh(gl-150000))*(a.value + b.value*gl + c.value*gl**2 + d.value*np.log(gl+1))

# Gráfico
plt.plot(glp_well2, np.transpose(qop_well2[11,:]),'o')
plt.plot(gl, qo_fitted)
plt.title("Well 2")
plt.xlabel("Gas injection")
plt.ylabel("Oil production")
plt.show()
plt.savefig("Well1.jpeg", dpi=300)

'''Fim de Estimação dos parametros do modelo'''

''' Finalizando contabilizador de tempo '''
TempoFim = timeit.default_timer()
print('Tempo de execução: %f' % (TempoFim - TempoInicio))

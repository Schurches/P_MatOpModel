#from flask import Flask
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import keras
from keras.models import load_model
global graph,model
import tensorflow as tf
import numpy as np

# Load model
graph = tf.get_default_graph()
model = load_model('NewPlayerNeuralNet.h5')
sc = joblib.load('scaler.joblib')
mediansDF = pd.read_csv('medians.csv')
med = mediansDF.groupby(['colegio','loID','grado'])['tiempo'].mean()

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


def sumEasyAndHardScores(array):
    """Recibe un arreglo 2xN con el historial de ejercicios fáciles y difíciles

    Parameters:
        
        array (array): Arreglo con historial de ejercicios

    Returns:
        
        [int,int]:Total de ejercicios fáciles y difíciles

   """     
    totalE = 0
    totalH = 0
    for i in range(0,len(array)):
        totalE = totalE+array[i][0]
        totalH = totalH+array[i][1]
    return [totalE,totalH]

#Sum bad, medium and hard categories rates to get %
def sumPerformanceCategories(array):
    """Recibe un arreglo 3xN con el historial de puntajes en cada categoría de desempeño

    Parameters:
        
        array (array): Arreglo con historial de porcentajes en cada categoría

    Returns:
        
        [int,int,int]:Puntaje total en cada categoría

   """
    totalB = 0
    totalM = 0
    totalG = 0
    for i in range(0,len(array)):
        totalB = totalB + array[i][0]
        totalM = totalM + array[i][1]
        totalG = totalG + array[i][2]
    return [totalB, totalM, totalG]


#General Binomial model
def binomialRate(binomialS, W, a):
    """Calcula el porcentaje de ejercicios fáciles y difíciles para la siguiente sesión de juego

    Parameters:
        
        binomialS (array): Arreglo con historial de ejercicios fáciles y difíciles
        W (int): Peso no-informativo del modelo binomial
        a (int): Valor inicial del modelo binomial

    Returns:
        
        array: Porcentaje de ejercicios fáciles y difíciles para la siguiente sesión

   """
    r1,r2 = sumEasyAndHardScores(binomialS)
    den = W+r1+r2
    S1 = (r1+W*a)/den
    S2 = (r2+W*a)/den
    scores = [S1,S2]
    scores = np.around(scores,decimals=2)
    return scores

#General Multinomial model
def multinomialRate(multinomialS, W, a):
     """Calcula la reputación del jugador en cada categoría de desempeño con base en su desempeño histórico

    Parameters:
        
        multinomialS (array): Arreglo con historial de reputación en cada categoría de desempeño
        W (int): Peso no-informativo del modelo multinomial
        a (int): Valor inicial del modelo multinomial

    Returns:
        
        array: Reputación para en cada categoría para la siguiente sesión de juego

   """
    r1,r2,r3 = sumPerformanceCategories(multinomialS)
    den = W+r1+r2+r3
    S1 = (r1+W*a) / den
    S2 = (r2+W*a) / den
    S3 = (r3+W*a) / den
    scores = [S1,S2,S3]
    scores = np.around(scores,decimals=2)
    return scores

#Get number of questiosn per percentage
def questionDistributions(intensity,n):
     """Calcula el número de ejercicios para cada dificultad"""
    N = []
    for i in range (0,len(intensity)):
        N.append(n*intensity[i])
    N = np.around(N)
    return N

#Get score from player performance
def giveScoreToPlayer(difficulty, time, ansChangedCount, isCorrect, LO, grade, isPublicOrPrivate):
     """Evalúa cómo le fue al jugador resolviendo un ejercicio en particular

    Parameters:
        
        difficulty (int): Dificultad del ejercicio
        time (float): Tiempo que demoró contestando
        ansChangedCount (int): Cuántas veces cambió entre opciones de respuesta
        isCorrect (int): Si contestó el ejercicio de manera correcta o incorrecta
        LO (int): Objetivo de aprendizaje asociado al ejercicio
        grade (int): Grado que cursa el jugador
        isPublicOrPrivate (int): Si el colegio del jugador es público o privado

    Returns:
        
        int: Puntaje para el ejercicio

   """
    medianTime = med[isPublicOrPrivate,LO,grade]+2
    if(time <= medianTime and isCorrect == 1):
        S = 20
    else:
        time = time-medianTime;
        bonus = difficulty+1
        S = ((20*bonus)*(1-time/medianTime)-2*(ansChangedCount-1))*isCorrect
    return S

#This returns the NEXT amount of easy and hard exercises
def rateRules1(nExercises, nEasy, nHard, scores, difficulty):
    """Varía la cantidad de ejercicios fáciles y difíciles para un objetivo de aprendizaje

    Parameters:
        
        nExercises (int): Cantidad de ejercicios 
        nEasy (int): Cantidad de ejercicios fáciles del objetivo
        nHard (int): Cantidad de ejercicios difíciles del objetivo
        scores (int): Puntajes en cada ejercicio
        difficulty (int): dificultad del ejercicio

    Returns:
        
        array: Nueva cantidad de ejercicios fáciles y difíciles para la categoría actual

   """
    easyCounter = nEasy;
    hardCounter = nHard;
    for i in range(0,len(scores)):
        if(difficulty[i] == 0):
            if(scores[i] >= 16):
                hardCounter=hardCounter+1
                easyCounter=easyCounter-1
            elif(scores[i] < 10 ):
                easyCounter=easyCounter+1
                hardCounter=hardCounter-1
            if(easyCounter < 0):
                easyCounter = 0
            elif(easyCounter > nExercises):
                easyCounter = nExercises
            if(hardCounter < 0):
                hardCounter = 0
            elif(hardCounter > nExercises):
                hardCounter = nExercises
        else:
            if(scores[i] >= 28):
                hardCounter=hardCounter+1
                easyCounter=easyCounter-1
            elif(scores[i] < 12 ):
                easyCounter=easyCounter+1
                hardCounter=hardCounter-1
            if(easyCounter < 0):
                easyCounter = 0
            elif(easyCounter > nExercises):
                easyCounter = nExercises
            if(hardCounter < 0):
                hardCounter = 0
            elif(hardCounter > nExercises):
                hardCounter = nExercises
    score = [easyCounter, hardCounter]
    return score;

#This increases the amount of % of exercises in each LO
def rateRules2(easyBinomialRates, exercises):
    """Varía los puntajes de reputación para cada categoría de desempeño de cada objetivo de aprendizaje

    Parameters:
        
        easyBinomialRates (array): Porcentaje de ejercicios fáciles en cada objetivo de aprendizaje
        exercises (array): Cantidad de ejercicios en cada objetivo de aprendizaje

    Returns:
        
        array: Reputaciones para cada categoría de desempeño

   """
    multinomialCounter = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for i in range(0,5):
        if(exercises[i] > 0):
            if(easyBinomialRates[i] >= 0.7):
                multinomialCounter[i][0] = multinomialCounter[i][0]+2
                if( i == 0 ):
                    multinomialCounter[i+1][0] = multinomialCounter[i+1][0]+1
                j = i-1
                while(j >= 0):
                    multinomialCounter[j][1] = multinomialCounter[j][1]+1
                    j = j-1
            elif(easyBinomialRates[i] >= 0.4 and easyBinomialRates[i] < 0.7):
                multinomialCounter[i][1] = multinomialCounter[i][1]+1
            else:
                multinomialCounter[i][2] = multinomialCounter[i][2]+2
                if(i < 4):
                    multinomialCounter[i+1][1] = multinomialCounter[i+1][1]+1
                j = i-1
                while(j >= 0):
                    multinomialCounter[j][2] = multinomialCounter[j][2]+1
                    j = j-1
    return multinomialCounter

#Apply aging factor to multinomial model
def applyMultinomialAging(multinomialHist,agingFactor):
    """ Reduce los puntajes históricos en cada categoría de desempeño """
    for i in range(0,len(multinomialHist)):
        multinomialHist[i][0] = multinomialHist[i][0]*agingFactor
        multinomialHist[i][1] = multinomialHist[i][1]*agingFactor
        multinomialHist[i][2] = multinomialHist[i][2]*agingFactor
    return np.around(multinomialHist,decimals=2)

#Apply aging factor to binomial model
def applyBinomialAging(binomialHist,agingFactor):
    """ Reduce los puntajes históricos en del modelo binomial """
    for i in range(0,len(binomialHist)):
        binomialHist[i][0] = binomialHist[i][0]*agingFactor
        binomialHist[i][1] = binomialHist[i][1]*agingFactor
    return np.around(binomialHist,decimals=2)

#Get next group of intensities
def calculateIntensity(multinomialPer):
    """Calcula el porcentaje de ejercicios para la siguiente sesión de juego en cada objetivo de aprendizaje

    Parameters:
        
        multinomialPer (array): Porcentaje de reputación para cada categoría de cada objetivo de aprendizaje

    Returns:
        
        array: Porcentaje de ejercicios para cada categoría

   """
    i = 0;
    perN = []
    while(i < 4 and multinomialPer[i][2] >= 0.8):
        perN.append(0)
        i = i+1;
    valuesN = []
    valuesN.append(1-multinomialPer[i][2])
    j = i+1
    k=1
    while(j < 5 and j <= i+2 and multinomialPer[j][0] <= 0.5):
       valuesN.append(1-multinomialPer[j][2])
       k=k+1
       j=j+1
    totalN = sum(valuesN)
    for j in range(0,len(valuesN)):
        perN.append(valuesN[j] / totalN)
        i=i+1
    for j in range(i,5):
        perN.append(0)
    return np.around(perN,decimals=2)

def obtainNextIntensity(nEasy, nHard, total, nSesion, binHist, binPer, bin_a, binAging, mulHist, mulPer, mult_a, mulAging, W, difficulty, times, titubeo, isCorrect, LO, grade, isPublicOrPrivate):
    """Calcula la cantidad de ejercicios para la siguiente sesión de juego

    Parameters: 
        
        nEasy (array): Cantidad de ejercicios fáciles para cada objetivo de aprendizaje \n
        nHard (array): Cantidad de ejercicios difíciles para cada objetivo de aprendizaje \n
        total (array): Total de ejercicios para cada objetivo de aprendizaje \n
        nSesion (int): Número de la sesión de juego \n
        binHist (array): Historial de cantidad de ejercicios fáciles y difíciles para cada objetivo de aprendizaje \n
        binPer (array): Historial de porcentajes de ejercicios fáciles y difíciles en cada objetivo de aprendizaje \n
        bin_a (float): Valor inicial para el modelo binomial \n
        binAging (float): Factor de envejecimiento del modelo binomial \n
        mulHist (array): Historial de reputaciones en cada categoría de desempeño para cada objetivo de aprendizaje \n
        mulPer (array): Historial de porcentajes de reputación para cada categoría de objetivo de aprendizaje \n
        mult_a (float): Valor inicial para el modelo multinomial \n
        mulAging (float): Factor de envejecimiento del modelo multinomial \n
        W (int): Peso no-informativo para los modelos de reputación \n
        difficulty (array): Dificultades de cada ejercicio contestado \n
        times (array): Tiempo tardado en contestar cada ejercicio \n
        titubeo (array): Cantidad de veces que alternó entre opciones de respuesta en cada ejercicio \n
        isCorrect (array): Si cada ejercicio fue resuelto de manera correcta o incorrecta \n
        LO (array): Objetivo de aprendizaje asociado a cada ejercicio \n
        grade (int): Grado que está cursando el jugador \n
        isPublicOrPrivate (int): Si el colegio del jugador es público o privado \n

    Returns:
        
        list: Historial de puntuaciones del modelo binomial \n
        list: Historiales de porcentajes del modelo binomial \n
        list: Historial de puntuaciones del modelo multinomial \n 
        list: Historiales de porcentajes del modelo multinomial \n
        list: Cantidad de ejercicios para cada objetivo de aprendizaje en la siguiente sesión \n

   """
    binomialEasyRatesPerLO = []
    #Calculating binomial scores
    if(nSesion % 5 == 0):
        for i in range (0,5):
            agedValues = applyBinomialAging(binHist[i],binAging)
            agedValues = agedValues.tolist()
            binHist[i] = agedValues
    n = 0
    for i in range (0,5):
        if(total[i]>0):
            easyExcercises, hardExcercises = nEasy[i], nHard[i]
            #Calculate scores
            scores = []
            diff = []
            for j in range(n,n+total[i]):
                scores.append(giveScoreToPlayer(difficulty[j],times[j],titubeo[j],isCorrect[j],LO[j],grade,isPublicOrPrivate))
                diff.append(difficulty[j])
            n=n+total[i]#Here goes calculus
            #Calculate scores
            binomialScores = rateRules1(total[i],easyExcercises,hardExcercises,scores,diff)
            binHist[i].append(binomialScores)
            binomialRates = binomialRate(binHist[i],W,bin_a)
            binPer[i] = binomialRates.tolist()
        binomialEasyRatesPerLO.append(binPer[i][0])

    #Calculating multinomial
    multinomialScores = rateRules2(binomialEasyRatesPerLO,total)
    for i in range(0,5):
        if(nSesion % 10 == 0):
            agedValues = applyMultinomialAging(mulHist[i],mulAging)
            agedValues = agedValues.tolist()
            mulHist[i] = agedValues
        mulHist[i].append(multinomialScores[i])
        mulPer[i] = multinomialRate(mulHist[i],W,mult_a).tolist()
    #Calculate intensities
    intensities = calculateIntensity(mulPer)
    return [binHist, binPer, mulHist, mulPer, intensities]

def obtainEasyAndHardCount(difficulties, LO):
    """Calcula la cantidad de ejercicios fáciles y difíciles para cada objetivo de aprendizaje

    Parameters:
        
        difficulties (array): Arreglo de dificultades fáciles y difíciles
        LO (array): Objetivo de aprendizaje asociado a cada dificultad

    Returns:
        
        [list]: Lista con el total de ejercicios, ejercicios fáciles y difíciles

   """
    nEasy = []
    nHard = []
    total = []
    for i in range (0,5):
        easyCount = 0
        hardCount = 0
        totalCount = 0
        for j in range(0,len(difficulties)):
            if(LO[j] == i):
                totalCount=totalCount+1
                if(difficulties[j] == 0):
                    easyCount=easyCount+1
                else:
                    hardCount=hardCount+1
        nEasy.append(easyCount)
        nHard.append(hardCount)
        total.append(totalCount)

    return [total, nEasy, nHard]

@app.route('/onNewPlayer', methods=['POST'])
def predict():
    """ Calcula una intensidad inicial de ejercicios para un nuevo jugador """
    playerData = request.get_json(force=True)
    d = {'edad': [playerData['edad']], 'escuela':[playerData['tipoEscuela']], 'genero': [playerData['genero']], 'grado': [playerData['grado']]}
    df = pd.DataFrame(data=d)
    X = sc.transform(df)
    with graph.as_default():
        intensities = model.predict(X)
        if(d['grado'][0] <= 3):
            total = intensities[0,0]+intensities[0,1]+intensities[0,2]
            intensities[0,0] = intensities[0,0]/total
            intensities[0,1] = intensities[0,1]/total
            intensities[0,2] = intensities[0,2]/total
            intensities[0,3] = 0
            intensities[0,4] = 0
        LO0 = "\"LOIN0\":["+str(intensities[0,0])+",1,0],"
        LO1 = "\"LOIN1\":["+str(intensities[0,1])+",1,0],"
        LO2 = "\"LOIN2\":["+str(intensities[0,2])+",1,0],"
        LO3 = "\"LOIN3\":["+str(intensities[0,3])+",1,0],"
        LO4 = "\"LOIN4\":["+str(intensities[0,4])+",1,0]"
        jsonFormatIntensities = "{"+LO0+LO1+LO2+LO3+LO4+"}"
        return jsonFormatIntensities

@app.route('/nextIntensity', methods=['POST'])
def onPerformanceReceived():
    """Calcula la cantidad de ejercicios con base en el desempeño del jugador en la última sesión de juego y considerando su historial """
    playerData = request.get_json(force=True)
    d = {'LOs': playerData['LO'],
         'grado': playerData['grado'],
         'sesion': playerData['sesion'],
         'binLO0': playerData['binLO0'],
         'binLO1': playerData['binLO1'],
         'binLO2': playerData['binLO2'],
         'binLO3': playerData['binLO3'],
         'binLO4': playerData['binLO4'],
         'binPer': playerData['binPer'],
         'mulLO0': playerData['mulLO0'],
         'mulLO1': playerData['mulLO1'],
         'mulLO2': playerData['mulLO2'],
         'mulLO3': playerData['mulLO3'],
         'mulLO4': playerData['mulLO4'],
         'mulPer': playerData['mulPer'],
         'tiempos': playerData['tiempos'],
         'titubeo': playerData['titubeo'],
         'isCorrect': playerData['correcto'],
         'colegio' : playerData['tipoEscuela'],
         'dificultades': playerData['dificultad']}

    n = len(d['isCorrect'])
    W = 2;
    bin_a = 0.5
    mult_a = 0.33
    bin_aging = 0.5
    mult_aging = 0.3
    #Variables
    sesion = d['sesion']
    grade = d['grado']
    isPublicOrPrivate = d['colegio']
    #Performance
    difficulty = d['dificultades']
    time = d['tiempos']
    ansChangedCount = d['titubeo']
    isCorrect = d['isCorrect']
    LOs = d['LOs']
    total, nEasy, nHard = obtainEasyAndHardCount(difficulty,LOs)
    #Binomial instances
    binomial_hist = [d['binLO0'],d['binLO1'],d['binLO2'],d['binLO3'],d['binLO4']]
    binomial_per = d['binPer']
    #Multinomial instances
    multinomial_hist = [d['mulLO0'],d['mulLO1'],d['mulLO2'],d['mulLO3'],d['mulLO4']]
    multinomial_per = d['mulPer']
    binomial_hist, binomial_per, multinomial_hist, multinomial_per, intensities = obtainNextIntensity(nEasy, nHard, total, sesion,
                        binomial_hist, binomial_per, bin_a, bin_aging, multinomial_hist, multinomial_per, mult_a, mult_aging,
                        W, difficulty, time, ansChangedCount, isCorrect, LOs, grade, isPublicOrPrivate)
    binJSON = "\"binLO0\":"+str(binomial_hist[0])+", \"binLO1\":"+str(binomial_hist[1])+", \"binLO2\":"+str(binomial_hist[2])+", \"binLO3\":"+str(binomial_hist[3])+", \"binLO4\":"+str(binomial_hist[4])+", \"binPer\":"+str(binomial_per)+", "
    mulJSON = "\"mulLO0\":"+str(multinomial_hist[0])+", \"mulLO1\":"+str(multinomial_hist[1])+", \"mulLO2\":"+str(multinomial_hist[2])+", \"mulLO3\":"+str(multinomial_hist[3])+", \"mulLO4\":"+str(multinomial_hist[4])+", \"mulPer\":"+str(multinomial_per)+", "
    intensityJSON = ""
    for i in range(0,5):
        intensityJSON = intensityJSON + "\"LOIN"+str(i)+"\":["+str(intensities[i])+","+str(binomial_per[i][0])+","+str(binomial_per[i][1])+"]"
        if(i<4):
            intensityJSON = intensityJSON + ", "
    jsonIntensityChild = "\"Intensities\": {"+intensityJSON+"}"
    return "{ "+binJSON+mulJSON+jsonIntensityChild+" }"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)

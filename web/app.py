# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:41:58 2019

@author: Steven
"""

from flask import Flask, request, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
import pandas as pd
import keras
from keras.models import load_model
import pickle
import sys
import tensorflow as tf
global graph,model
import numpy as np


# Flask
app = Flask(__name__)

# Load model
graph = tf.get_default_graph()
model = load_model('NewPlayerNeuralNet.h5')
sc = joblib.load('scaler.joblib')
mediansDF = pd.read_csv('medians.csv')
med = mediansDF.groupby(['colegio','loID','grado'])['tiempo'].mean()


#Sum binomial rates to get easy and hard %
def sumEasyAndHardScores(array):
    totalE = 0
    totalH = 0
    for i in range(0,len(array)):
        totalE = totalE+array[i][0]
        totalH = totalH+array[i][1]
    return [totalE,totalH]

#Sum bad, medium and hard categories rates to get %
def sumPerformanceCategories(array):
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
    r1,r2 = sumEasyAndHardScores(binomialS)
    den = W+r1+r2
    S1 = (r1+W*a)/den
    S2 = (r2+W*a)/den
    scores = [S1,S2]
    scores = np.around(scores,decimals=2)
    return scores

#General Multinomial model
def multinomialRate(multinomialS, W, a):
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
    N = []
    for i in range (0,len(intensity)):
        N.append(n*intensity[i])
        
    N = np.around(N)
    return N

#Get score from player performance
def giveScoreToPlayer(difficulty, time, ansChangedCount, isCorrect, LO, grade, isPublicOrPrivate):
    medianTime = med[isPublicOrPrivate,LO,grade]+2
    bonus = difficulty+1
    S = ((20*bonus)*(1-time/medianTime)-2*(ansChangedCount-1))*isCorrect
    return S    

#This returns the NEXT amount of easy and hard exercises
def rateRules1(nExercises, nEasy, nHard, scores, difficulty):
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
    for i in range(0,len(multinomialHist)):
        multinomialHist[i][0] = multinomialHist[i][0]*agingFactor
        multinomialHist[i][1] = multinomialHist[i][1]*agingFactor
        multinomialHist[i][2] = multinomialHist[i][2]*agingFactor
    return np.around(multinomialHist,decimals=2)

#Apply aging factor to binomial model
def applyBinomialAging(binomialHist,agingFactor):
    for i in range(0,len(binomialHist)):
        binomialHist[i][0] = binomialHist[i][0]*agingFactor
        binomialHist[i][1] = binomialHist[i][1]*agingFactor
    return np.around(binomialHist,decimals=2)

#Get next group of intensities
def calculateIntensity(multinomialPer):
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
    mult_a = 0.2
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


@app.route('/onNewPlayer', methods=['GET'])
def onWelcome():
    return 'hola'


if __name__ == '__server__':
    app.run(host="localhost", port=5000)

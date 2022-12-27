# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:20:04 2019

@author: ke0015
"""

#%reset -f
import time
start_time = time.time()
import csv
import numpy as np
# import pandas as pd
from sys import platform
from pandas import read_csv
import RNN_Predict as pr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def debug_signal_handler(signal, frame):  #pause execution with ctl+c, resume with c
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)

#%% -init-
maxMCNdB = 19.57        # maximum for ModCod
totalSpecEff = 0
switchingThreshold = 10 # dB
trainingPctge = 0.9     # Training percentage
# numHiddenUnits = 200    # Number of hidden units
# maxEpochs = 250         # maximum number of epochs
# alpha = 0.005           # learning rate
learning_rate = 0.001
num_periods = 20  #20
f_horizon = 1  # forecast horizon, one period into the future
epochs = 1000 #1000
    #learning_rate = 0.001
hidden = 100 #100
predictAheadMins = 6
minPredictAheadMins, maxPredictAheadMins = 3, 6     # minutes

siteIdxNFade, specEffData, switchData = [], [], []

#%% load dataset   
series = read_csv('./MayJul17Cured.csv')
ModCod = read_csv('./ModCod.csv')

totalAttenData = series.values

#%%  Correlation between data
#fadeData = series[['Chilbolton','Edinburgh']]
#correlation = fadeData.corr(method='pearson')
#print(correlation)
#plt.scatter(series.loc[:, 'Chilbolton'], series.loc[:, 'Edinburgh'])

#%% Switching
if totalAttenData[0][1] < totalAttenData[0][2]:  # Initial starting site
    rxSite = 1  # Chilbolton
else:
    rxSite = 2  # Scotland

predictAheadData = predictAheadMins * 60    # seconds
trainData = int(predictAheadData * trainingPctge / (1 - trainingPctge))

#%% Loop for minutes
for predictAheadMins in range(minPredictAheadMins, maxPredictAheadMins):
    switchData = []
    sameSiteCount, switchCount, totalSpecEff = 1, 1, 0
    
    for k in range(0, len(totalAttenData)):
        snrdB = maxMCNdB - totalAttenData[k][rxSite] + 0.5      # snr
        specEffIdx = ModCod[(ModCod['SNRdB-upper'] >= snrdB) & (ModCod['SNRdB-lower'] < snrdB)].values
        specEff = specEffIdx[0][4]
        totalSpecEff += specEff             # total spectral efficiency
        
        if k > trainData:
            if totalAttenData[k][rxSite] > switchingThreshold:
                siteData = totalAttenData[k - trainData : k + predictAheadData,:]
                siteData = siteData[:,rxSite]
                YPred = pr.RNN_Predict(siteData, epochs, learning_rate, hidden, num_periods, f_horizon)                     #siteData, learning_rate, trainingPctge)
                #YPred = RNN_Predict(siteData, learning_rate, trainingPctge)
                if np.amax(YPred) > totalAttenData[k][rxSite]:   # check if max(predictedFade) > currentFadeLevel
                    altSite = (rxSite % 2) + 1      # Alternate site

                if totalAttenData[k][altSite] < totalAttenData[k][rxSite]:   # check if fade(otherSite) < fade(currentSite)
                    siteData = totalAttenData[k - trainData : k + predictAheadData,:]
                    siteData = siteData[:,rxSite]
                    YPred = pr.RNN_Predict(siteData, epochs, learning_rate, hidden, num_periods, f_horizon)                     #siteData, learning_rate, trainingPctge)
                    #YPred = RNN_Predict(siteData, learning_rate, trainingPctge)
                    if np.amax(YPred) > totalAttenData[k][rxSite]:    # check if max(predictedFade) > currentFadeLevel in the alt site
                        rxSite = altSite   # Switch to the alternate site
        
        rxFade = totalAttenData[k][rxSite]
        
        if k > 1:
            if totalAttenData[k-2][0] == rxSite:
                sameSiteCount += 1
            else:
                switchData.append([k, totalAttenData[k][0], rxSite, sameSiteCount, rxFade, totalAttenData[k][1], totalAttenData[k][2]])
                switchCount += 1
                sameSiteCount = 1

        siteIdxNFade.append([rxSite, rxFade])

        if k % 10000 == 0:
            if rxSite == 1:
                siteLoc = "Chilbolton"
            else:
                siteLoc = "Edinburgh"

            print("\nDateTime={}, rxSite={}, Mins={:,}/{:,} [{:,}/{:,}]\n".format(totalAttenData[k][0].strip(), siteLoc, predictAheadMins, maxPredictAheadMins, k, len(totalAttenData)))

    aveSpecEff = totalSpecEff / len(totalAttenData)
    print("\n\t Average Spectral Efficiency (FER) = {:5.2f}\n".format(aveSpecEff))
    
    #%% Save results
    newAttenData = np.concatenate((totalAttenData, siteIdxNFade), axis=1)
    
    specEffData.append([predictAheadMins, aveSpecEff])

    if platform == 'win32':
        SwitchDataFile = "S:\Research\AlphasatData\Results\SwDataFilePred2#"+"{:,}".format(predictAheadMins)+"mins.csv"
        specEffDataFile = "S:\Research\AlphasatData\Results\SpEffDataFilePred2.csv"
    elif platform == 'linux' or platform == 'linux2':
        SwitchDataFile = "/vol/research/AlphasatData/Results/SwDataFilePred2#"+"{:,}".format(predictAheadMins)+"mins.csv"
        specEffDataFile = "/vol/research/AlphasatData/Results/SpEffDataFilePred2.csv"

    with open(SwitchDataFile, 'w', newline='') as outfile:
        csv.writer(outfile).writerows(switchData)
    
    try:
        with open(specEffDataFile, 'w', newline='') as outfile:
            csv.writer(outfile).writerow(specEffData)
    except:
        with open(specEffDataFile, 'a', newline='') as outfile:
            csv.writer(outfile).writerow(specEffData)
            
#%%########################
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Rain Fades and Switching for Chilbolton and Edinburgh", fontsize=14)  #%%s current_date));
plt.plot([row[1] for row in siteIdxNFade], "b", markersize=10, label="rxFade")  # 
plt.plot([row[0] for row in siteIdxNFade], "r", markersize=10, label="Site (1 = Chilbolton, 2 = Edinburgh)",linewidth=2.0)  # rxSite
plt.xlim(0, len(siteIdxNFade))
plt.legend(loc="best")
plt.xlabel("Time (seconds)")
plt.ylabel("Rain fade (dB)")
plt.grid(which='major')
plt.show()

elapsed_time = time.time() - start_time   #toc
print("elapsedTime : {:6.3f} s".format(elapsed_time))
print('Fin!')
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:10:04 2021

@author: Lenovo
"""
import numpy as np
import random as rd
from random import gauss
#import matplotlib.pyplot as plt

def initial_pop(func,dim,bounds,popsize):
    initialpop = np.zeros((popsize,dim))
    a = bounds[0]
    b = bounds[1]
    for i in range(popsize):
        for j in range(dim):
            initialpop[i,j] = rd.uniform(a,b)
    return initialpop


def selection(alist,point):
    roulette = np.zeros((len(alist),2))
    roulette[0,0] = 0
    roulette[0,1] = alist[0]
    for i in range(1,len(alist)):
        roulette[i,0] = roulette[i-1,1] 
        roulette[i,1] = roulette[i,0] + alist[i]
    ans = 0
    for i in range(len(alist)):
        if (roulette[i,0] - point)*(roulette[i,1] - point) > 0:
            ans = ans + 1
        else:
            break
    return ans
            
#defining a general GA algorithim 
def GA(func,dim,bounds,popsize,pops,pcrossover,pmut,selectionindex):
    #fitness evaluation and storage
    functionalevals = []
    
    for i in range(popsize):
        functionalevals.append((func(pops[i],dim),i))
    functionalevals.sort(key=lambda tup: tup[0],reverse = True)
    ranking = np.zeros(popsize)
    
    for j in range(popsize):
        ranking[functionalevals[j][1]] = j+1
    maximum = rd.uniform(1.1,2.0)
    fitness = []
    
    for k in range(popsize):
        a = (2 - maximum) + (2*(maximum-1)*((ranking[k] - 1)/(popsize - 1)))
        fitness.append(a)
    elite = max(fitness)
    elite_index = fitness.index(elite)
    elite_value = func(pops[elite_index],dim)
    elite_loc = pops[elite_index]
    
    #selection
    SUM = sum(fitness)
    selectedpop = []
    if selectionindex == 0:
        for i in range(int(pcrossover*popsize)):
            dot = rd.uniform(0,SUM)
            selectedpop.append(pops[selection(fitness,dot)])
        selectedpop = np.asarray(selectedpop)
    elif selectionindex == 1:
        for i in range(int(pcrossover*popsize)):
            dot = rd.uniform(0,SUM)
            selectedpop.append(pops[selection(ranking,dot)])
        selectedpop = np.asarray(selectedpop)
    #crossover
    nextgen = []
    listforcross = np.arange(0,1,(1/int((pcrossover*popsize))))    
    for i in range(0,popsize,2):
        seed1 = rd.random()
        seed2 = rd.random()
        seed  = rd.random()
        parentA = selectedpop[selection(listforcross,seed1)]
        parentB = selectedpop[selection(listforcross,seed2)]
        nextgen.append(seed*parentA + (1-seed)*parentB)
        nextgen.append(seed*parentB + (1-seed)*parentA)
    nextgen = np.asarray(nextgen)
    #mutation
    for i in range(popsize):
        ptr = rd.random()
        if ptr < pmut:
            for l in range(dim):
                nextgen[i,l] = nextgen[i,l] + gauss(0,1)
    return [nextgen,elite_value,elite_loc]

def final_GA(func,dim,bounds,popsize,pcrossover,pmut,Niter,selectionindex):
    gen0 = initial_pop(func,dim,bounds,popsize)
    gens = [gen0]
    highest = []
    best_points = []
    for i in range(1,Niter):
        gens.append(GA(func,dim,bounds,popsize,gens[i-1],pcrossover,pmut,selectionindex)[0])
        highest.append(GA(func,dim,bounds,popsize,gens[i-1],pcrossover,pmut,selectionindex)[1])
        best_points.append(GA(func,dim,bounds,popsize,gens[i-1],pcrossover,pmut,selectionindex)[2])
    return [gens[Niter-1],highest,best_points]

def parabola(x,d):
    return x[0]**2 + x[1]**2

#defining different functions
def bohachevsky(x,d):
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) +0.7

def rotated_hyper_ellipsoidal(x,d):
    ans = 0
    for i in range(d):
        for j in range(i+1):
            ans = ans + x[j]**2
    return ans

def powell(x,d):
    ans = 0
    for i in range(d):
        ans = ans + (abs(x[i]))**(i+1)
    return ans

def dixon_price(x,d):
    ans = (x[0] - 1)**2
    for i in range(1,d):
        ans = ans + i*((2*x[i]**2) - x[i-1])**2
    return ans

def brown(x,d):
    ans = 0
    for i in range(d-1):
        ans = ans + (x[i]**2)**(x[i+1]**2 + 1) + (x[i+1]**2)**(x[i]**2 + 1)
    return ans

def ackleys(x,d):
    a =20
    b=0.2
    c= 2*np.pi
    s1 = 0
    s2 = 0
    for i in range(d):
        s1 = s1 + x[i]**2
        s2 = s2 + np.cos(c*x[i])
    return   -a*np.exp(-b*np.sqrt(s1/d))  - np.exp(s2/d) + a + np.exp(1) 

def rosenbrock(x,d):
    ans = 0
    for i in range(d-1):
        ans = ans + 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return ans

def levy(x,d):
    w = []
    for i in range(d):
        w.append(1 + (x[i]-1)/4)
    term = (np.sin(np.pi*w[0]))**2 + (((w[d-1] - 1)**2)*(1 + (np.sin(2*np.pi*w[d-1]))**2))
    ans = 0
    for i in range(d-1):
        ans = ans + ((w[i] - 1)*(1 + 10*(np.sin(np.pi*w[i] + 1))**2))
    return term + ans

def rastrigin(x,d):
    ans = 10*d
    for i in range(1,d):
        ans = ans + x[i]**2 -  10*np.cos(2*np.pi*x[i])
    return ans


def bohachevsky_multimod(x,d):
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1]) +0.3


def absolute(x):
    return [abs(number) for number in x]



for i in range(5):
    GAresults = final_GA(rosenbrock,2,[-5.12,5.12],40,0.8,0.1,1000,1)
    #plt.plot(final_GA(bohachevsky,2,[-5.12,5.12],25,0.8,0.1,1000,0)[1])  
    #print(min(abs(final_GA(bohachevsky,2,[-5.12,5.12],25,0.8,0.1,1000,0)[1])))     
    best_value = (min(absolute(GAresults[1]))) 
    best_index = (absolute(GAresults[1])).index(best_value)
    best_point = GAresults[2][best_index]
    print(best_value , (best_point[0],best_point[1]))
  
          
    
    
    


#!/usr/bin/env python2

import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

m = np.loadtxt('/home/wizkraft/openface-master/demos/matrix_dabangg.txt', usecols=range(128))

print m.shape
n = len(m[:,127])


'''
R[i] stores the frequency of appearance of Person i
N[i] stores the centre/first image of Person i
M[i] stores the number of the person assigned to ith image

''' 
M = np.zeros(n+2)
N = np.zeros(n+1)
R = np.zeros(n+1)

#initial = int(n/2)
initial = 0
M[initial]=1
N[1]=1
k=1


for i in range(initial,n-1):
    #statinfo = os.stat("/home/wizkraft/Desktop/Faces/Thor/"+str(i)+".jpg")
    #if statinfo.st_size < 10000:
    #    continue
    d = m[i,:] - m[i+1,:]
    if np.dot(d,d) < 1:
        M[i+1]=M[i]
    else:
        min = 5
        index = 1
        for j in range(1,k+1):
            d = m[int(N[j]),:] - m[i+1,:]
            if np.dot(d,d) < min:
                min = np.dot(d,d)
                index = int(N[j])
        if min < 0.95:
            M[i+1] = M[index]
        else:
            M[i+1]=k+1
            N[k+1]=i+1
            k=k+1

s = k   # number of different characters in the movie

for i in range(1,n):
    R[int(M[i])] = R[int(M[i])]+1


max1 = 0
max2 = 0
max3 = 0
max4 = 0

index1 = 0
index2 = 0
index3 = 0
index4 = 0

for i in range(1,s+1):
    print R[i],N[i]
    if max1 < R[i]:
        max1=R[i]
        index1 = i

for i in range(1,s+1):
    if i==index1:
        continue
    if max2 < R[i]:
        max2=R[i]
        index2 = i

for i in range(1,s+1):
    if i == index1 or i == index2:
        continue
    if max3 < R[i]:
        max3=R[i]
        index3 = i

for i in range(1,s+1):
    if i==index1 or i==index2 or i==index3:
        continue
    if max4 < R[i]:
        max4=R[i]
        index4 = i

print N[int(index1)]
print N[int(index2)]
print N[int(index3)]
print N[int(index4)]


frequencies = [0] * s
i=0

for i in range(0,s-1):
    frequencies[i] = int(R[i+1])

alphab = []
calc = 1
while int(calc) <= s:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title('Thor')
pyplot.xlabel('Person Number')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, frequencies, width, color='r')

plt.show()

t = np.loadtxt('/home/wizkraft/Desktop/Faces/Dabangg/images.txt', usecols=range(1))

noi = 0 #number of intervals
while(t[noi]>0):
    noi = noi+1

t[noi] = n+1

a1 = [0]*noi
a2 = [0]*noi
a3 = [0]*noi
a4 = [0]*noi


for i in range(0,noi) :
    for j in range(int(t[i]),int(t[i+1])):
        if M[j]==index1:
            a1[i]=a1[i]+1
        if M[j]==index2:
            a2[i]=a2[i]+1
        if M[j]==index3:
            a3[i]=a3[i]+1
        if M[j]==index4:
            a4[i]=a4[i]+1

print a1
print a2
print a3
print a4

print("--- %s seconds ---" % (time.time() - start))

alphab = []
calc = 1
while int(calc) <= noi:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title('Actor 1')
pyplot.xlabel('Time Interval')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, a1, width, color='r')

plt.show()

alphab = []
calc = 1
while int(calc) <= noi:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title('Actor 2')
pyplot.xlabel('Time Interval')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, a2, width, color='r')

plt.show()

# edit in line 21 to upload desired matrix
# edit in line 153 to upload the images matrix specifying time interval
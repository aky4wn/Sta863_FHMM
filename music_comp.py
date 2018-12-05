import numpy as np
import pandas as pd
import csv
from numpy import linspace,exp
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import seaborn as sns

import scipy 
import editdistance
import sklearn.metrics
import statsmodels.api as sm
## Scripts containing inference algorithms
from BaumWelch import *
from BaumWelchLR import *
from TVAR import *

## Helper functions for working with note pitch representation

## Convert from pitch representation (integers 0-127) to integers (0-max)
## x is the input vector of notes and code is a vector of the unique pitches in x
def encode(x, code):
    output = np.array([int(np.where(code == x[i])[0]) for i in range(0, len(x))])
    return output


## Reverses the function encode
## x is the vector of pitches to decode and code is a vector of the unique pitches in x before it was encoded
def decode(x, code):
    output = np.zeros(len(x))
    for i in range(0, len(x)):
        output[i] = code[x[i]]
    return output

## Function to convert the values in array to the nearest values in the array value
## Used to convert continues TVAR generated pitches to closest integer values for MIDI representation
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


####################### 
## Metrics Functions ##
#######################

## Function to create a matrix of notes and times representing the input piece as a matrix
## time is the time steps (integers) at which a note occurs
## notes are the note pitches (integers 0-127) where each note is "turned on" and "turned off"
## velocity is the note velocity (i.e. volume) at each time step
## measures is the number measures in the original piece
## min_note is the length of the shortest note in the original piece, using same scale as time array above
## num is the number of quarter notes in a measure (i.e. represents the numerator of time signature of the input piece)
## met_mat output is a matrix where columns correspond to the time stamp of notes, one column for each min_note
##    duration for the entire piece and the rows are the note pitches, values in the matrix are 1 for the duration
##    of a note when it is played and 0 when it is not played
def create_matrix(time, notes, velocity, measures, min_note, num):
    met_mat = pd.DataFrame(np.zeros(shape = (len(np.unique(notes)), int(measures)*num), dtype = int))
    met_mat.index = np.unique(notes)[::-1]
    met_mat.columns = np.arange(0, min_note*num*measures, min_note)[:int(measures)*num]
    max_time = met_mat.columns[-1]
    for i in np.unique(notes):
        on = time[np.intersect1d(np.where(notes == i), np.where(velocity > 0) )]
        off = time[np.intersect1d(np.where(notes == i), np.where(velocity == 0) )]
        if len(off) % 2 !=0 or len(on) %2 !=0:
            off = np.append(off, max_time)
        for j in range(len(on)):
            met_mat.loc[i, on[j]:off[j]] = 1
    return(met_mat)

## Function to calculate the musical metrics of generated pieces
## met_mat is the output from create_matrix()
## harmonic ints is a vector of length 12 corresponding to the counts of each type of harmonic interval in the piece considered
## melodic ints is a vector of length 12 corresponding to the counts of each type of melodic interval in the piece considered
## percentage is a vector of length 6 containing the percentage of perfect harmonic intervals, the percentage of imperfect 
##           consonant harmonic intervals, the percentage of dissonant harmonic intervals, the percentage of perfect melodic
##           intervals, the percentage of imperfect consonant melodic intervals and the percentage of dissonant melodic intervals
def musical_metrics(met_mat):
    perfect = np.array([0,5,7])
    imperfect = np.array([3,4,8,9])
    dissonant = np.array([1,2,6,10,11])
    major_scale = np.array([2,2,1,2,2,2,1])
    harmonic_ints = np.zeros(12)
    c = 0
    max_notes = np.max(np.sum(met_mat, axis = 0))
    melodic = np.zeros(shape = (max_notes, len(met_mat.columns)))
    for col in met_mat.columns:
        chord = np.array(met_mat.index[np.where(met_mat[col] == 1)[0]])[::-1]
        if len(chord) > 0:
            intervals = np.diff(chord)
            intervals[intervals >= 12] = intervals[intervals >= 12] % 12
            harmonic_ints[intervals.astype(int)] +=1
            melodic[:len(chord), c] = chord
            c+= 1
    melodic_ints = list()
    for t in range(melodic.shape[1] - 1):
        m1 = melodic[melodic[:,t] > 0, t]
        m2 = melodic[melodic[:,t+1] > 0, t+1]
        melodic_ints.append(np.unique([abs(i-j) %12 for i in m1 for j in m2]))

    u = np.unique(np.hstack(melodic_ints), return_counts=True)
    m_ints = np.zeros(12)
    m_ints[u[0].astype(int)] = u[1].astype(int)

    h_total = np.sum(harmonic_ints)
    m_total = np.sum(m_ints)

    h_per = np.sum(harmonic_ints[perfect])/h_total
    h_imp = np.sum(harmonic_ints[imperfect])/h_total
    h_dis = np.sum(harmonic_ints[dissonant])/h_total

    m_per = np.sum(m_ints[perfect])/m_total
    m_imp = np.sum(m_ints[imperfect])/m_total
    m_dis = np.sum(m_ints[dissonant])/m_total

    percentage = np.array([h_per, h_imp, h_dis, m_per, m_imp, m_dis])

    return(harmonic_ints, m_ints, percentage)

## Calculate the empirical entropy of the input data and output as a vector in entropy
def ent(data):
    p_data= np.unique(data, return_counts = True)[1]/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy


## Function to compare an original piece to a generated piece and calculate metrics
## old_notes is a vector of the original piece's note pitches
## new_notes is a vector of the generated piece's note pitches
## Returns the empirical entropy, mutual information and edit distance between the original piece and the new, generated piece
##        also returns the count of unique notes in the generated piece, normalized by the total number of notes
def originality_metrics_comparison(old_notes, new_notes):
    # Calculate entropy
    entropy = ent(new_notes)

    # Calculate edit distance
    edit_dist = editdistance.eval(old_notes, new_notes)/len(old_notes)

    # Calculate mutual info
    mutual_info = sklearn.metrics.mutual_info_score(old_notes, new_notes)

    k = len(np.unique(old_notes))
    possibleNotes = np.unique(old_notes)    
    # Calculate note counts
    unique_new_notes, note_counts = np.unique(new_notes, return_counts = True)

    if len(unique_new_notes) != k:
        add_notes = list(set(possibleNotes) - set(unique_new_notes))
        for i in add_notes:
            if np.where(possibleNotes == i)[0] > len(note_counts):
                note_counts = np.append(note_counts, np.where(possibleNotes == i)[0], 0)
            else:
                note_counts = np.insert(note_counts, np.where(possibleNotes == i)[0], 0)
    note_counts = note_counts/len(old_notes)
    return(entropy, mutual_info, edit_dist, note_counts)


## Function to calculate the ACF and PACF out to lag 40
## new_ntoes is the input vector of note pitches
## note_acf is a vector of length 41 of the ACF values and note_pacf is a vector of length 41 of the PACF values
def time_metrics(new_notes):   
    #Calculate ACF/PACF out to lag 40
    note_acf = sm.tsa.stattools.acf(new_notes)
    try:
        note_pacf = sm.tsa.stattools.pacf(new_notes)
    except np.linalg.linalg.LinAlgError as err:
        note_pacf = sm.tsa.stattools.pacf(new_notes)

    return(note_acf, note_pacf)


## Function to calculate all metrics
## time is the time steps (integers) at which a note occurs
## notes are the note pitches (integers 0-127) where each note is "turned on" and "turned off"
## velocity is the note velocity (i.e. volume) at each time step
## measures is the number measures in the original piece
## min_note is the length of the shortest note in the original piece, using same scale as time array above
## num is the number of quarter notes in a measure (i.e. represents the numerator of time signature of the input piece)
## output is a vector of the calculated metrics:
##        entropy is the empirical entropy of new_notes
##        mutual_info is the mutual information between old_notes and new_notes
##        edit_dist is the edit distance between old_notes and new_notes
##        harmonic_ints is a vector of length 12 of the count of harmonic intervals of each type in new_notes
##        melodic_ints is a vector of length 12 of the count of melodic intervals of each type in new_notes
##        percentage is a vector of length 6 containing the percentage of perfect harmonic intervals, the percentage of imperfect 
##           consonant harmonic intervals, the percentage of dissonant harmonic intervals, the percentage of perfect melodic
##           intervals, the percentage of imperfect consonant melodic intervals and the percentage of dissonant melodic intervals      
##       note_counts is a vector of length equal to the number of unique pitches in old_notes with a normalized count of pitches
##           in new_notes
##       note_acf is a vector of length 41 with the acf of new_notes
##       note_pacf is a vector of length 41 with the pacf of new_notes
## Note: time is the same for old_notes and new_notes, as this is not changed between the original and generated pieces
##  (likewise, measures, min_note and num are the same for old_notes and new_notes)
def calc_metrics(time, old_notes, new_notes, velocity, measures, min_note, num):
    met_mat = create_matrix(time, new_notes, velocity, measures, min_note, num)
    harmonic_ints, m_ints, percentage = musical_metrics(met_mat)
    entropy, mutual_info, edit_dist, note_counts = originality_metrics_comparison(old_notes, new_notes)
    note_acf, note_pacf = time_metrics(new_notes)
    return(np.hstack((np.array([entropy, mutual_info, edit_dist]), 
           harmonic_ints, m_ints, percentage, note_counts, note_acf, note_pacf)))



## Function to generate new pieces from the HMM, 2-HMM, 3-HMM, LR-HMM, 2LR-HMM, and 3-LR HMM
## n is the length of the original and generated piece
## pi is the learned initial distribution
## phi is the learned emission distribution
## Tmat is the learned transition matrix
## T2mat is the learned second order transition matrix (if applicable)
## T3mat is the learned third order transition matrix (if applicable)
## code is the unique note pitches occurring in the original piece
## model is the model order which the input parameters correspond to, either "first_order", "second_order" or "third_order"
## Outputs: output is the note pitches of the generated pieces (array of length n), z are the generated hidden states 
##          (vector of length n)
def hmm(n, pi, phi, Tmat, T2mat, T3mat, code, model):
    m = Tmat.shape[0]
    k = phi.shape[1]
    zstates = np.arange(0, m, dtype = int)
    xstates = np.arange(0, k, dtype = int)
    z = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z[0] = np.random.choice(zstates, size = 1, p = pi)
    if model == 'first_order':
        for j in range(1, n):
            z[j] = np.random.choice(zstates, size = 1, p = Tmat[z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size = 1, p = phi[z[i], :])
     
    if model == 'second_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        for j in range(2, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T2mat[z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    if model == 'third_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        z[2] = np.random.choice(zstates, size = 1,  p = T2mat[z[0],z[1], :])
        for j in range(3, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T3mat[z[j-3],z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    output = decode(x, code)
    return (output, z)

####################
## Pre-Processing ##
####################

## Function to pre-process input CSV of original song into form that can be used for modeling and metrics
## Assumes original piece's MIDI file has been converted to a CSV using http://www.fourmilab.ch/webtools/midicsv/#midicsv.5
## input_filename = name of original csv
## output_filename = name of new csv to save generated piece to
## Outputs:
##         quarter_note = number of time steps corresponding to 1 quarter note
##         num = numerator in key signature
##         denom = denominator in key signature
##         key = key signature of piece, integer between -7 and 7 where 0 is C Major
##         measures = number of measures in input piece
##         time = vector of time stamps at which notes occurr
##         notes = vector of note pitches (integers 0-127)
##         velocity = "volume" of each note pitch, 0 = note off, length of time is the same as length of notes and velocity
##         song = pandas dataframe to use for output generated pieces, retains formatting expected by MIDI-CSV
##         song.index = index of original song dataframe

## See http://www.fourmilab.ch/webtools/midicsv/#midicsv.5 for a discussion of MIDI and CSV formats

class pre_process(object):
    def __init__(self, input_filename, min_note):
        self.input_filename = input_filename
        self.min_note = min_note
      
    
    def read_process(self):
        with open(self.input_filename,encoding = "ISO-8859-1") as fd:
            reader=csv.reader(fd)
            rows= [row for idx, row in enumerate(reader)]
        song = pd.DataFrame(rows)
        r,c = np.where(song == ' Header')
        quarter_note = song.iloc[r,5].values.astype(int)[0]
        r, c = np.where(song == ' Time_signature')
        num = song.iloc[r, 3].values.astype(int)[0]
        denom = song.iloc[r, 4].values.astype(int)[0]**2
        try:
            r, c = np.where(song == ' Key_signature')
            key = song.iloc[r,3].values.astype(int)[0]
        except:
            key = None
        
        song_model = song.loc[song.iloc[:,0] == np.max(song.iloc[:,0])]
        song_model = song_model[song_model.iloc[:, 2].isin([' Note_on_c', ' Note_off_c'])]
        time = np.array(song_model.iloc[:,1]).astype(int)
        notes = np.array(song_model.iloc[:,4]).astype(int)
        velocity = np.array(song_model.iloc[:,5]).astype(int)
        measures = np.round(np.max(time)/quarter_note)/num
        min_note = quarter_note
        actual = np.arange(0, min_note*measures*num, min_note).astype(int) 
        time = np.array([find_nearest(actual, time[i]) for i in range(len(time))]).astype(int)
        return(quarter_note, num, denom, key, measures, time, notes, velocity, song, song_model.index)

    
##############    
## Velocity ##
##############

## newNotes = vector of note pitches of new, generated piece
## velocity = velocity of original piece
## newVelocities = velocities for newNotes, with 0s appropriately filled in and spline values for other non-0 values

def find_vel(newNotes, velocity):
    # Use splines to interpolate the velocities
    newVelocities = np.zeros(len(newNotes))
    y = velocity[np.nonzero(velocity)]
    indicies = []
    for i in np.unique(newNotes):
        indicies.append(np.where(newNotes == i)[0][::2])  ## set every other pitch occurrence to 0 (turn off)

    unlist = [item for sublist in indicies for item in sublist]
    unlist.sort()
    X = np.array(range(0,len(y)))
    s = UnivariateSpline(X, y, s=300) #750
    xs = np.linspace(0, len(y), len(unlist), endpoint = True)
    ys = s(xs)    
    newVelocities[np.array(unlist)] = np.round(ys).astype(int)
    #Fix entries that are too small or too large due to spline overfitting
    newVelocities[np.where(newVelocities < 0)[0]] = y[-1]
    newVelocities = newVelocities.astype(int)    

    return(newVelocities)

##########################
## Inference Algorithms ##
##########################

## Existing Models

def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    else:
        b = np.max(a)
        return(b + np.log(np.sum(np.exp(a-b))))


def pForward(g, x):
    pXf = logSumExp(g[len(x)-1,:])
    return(pXf)
def indicator(a, b):
    if a == b:
        return 1
    else:
        return 0
def pForwardARHMM(g, x):
    pXf = logSumExp(g[len(x)-1,:, :])
    return(pXf)

    
def forwardAlg(n, m, k, pi, Tmat, phi, x):
    g = np.zeros((n,m))
    for i in range(0,m):
        g[0,i] = (pi[i]) + (phi[i, x[0]])
    
    for j in range(1, n):
        for l in range(0, m):
            g[j,l] = logSumExp(np.asarray(g[j-1, :])+np.asarray(Tmat[:,l])+(phi[l,x[j]]))
    return(g)

def backwardAlg(n, m, k, pi, Tmat, phi, x):
    r = np.zeros((n,m))
    for j in range(n-2, -1, -1):
        for l in range(0, m):
            r[j, l] = logSumExp(np.asarray(r[j+1,: ]) + np.asarray(Tmat[l,:]) + phi[:, x[j+1]])
    
    return(r)

def Viterbi(n, m, k, pi, Tmat, phi, x):
    f = np.zeros(shape = (n,m))
    alpha = np.zeros(shape = (n,m))
    zStar = np.zeros(n)
   
    for t in range(0, n):
        for i in range(0,m):
            if t == 0:
                f[0, i] = pi[i] + phi[i, x[0]]
            else:
                u = np.asarray(f[t-1, :]) + np.asarray(Tmat[:, i]) + phi[i, x[t]]
                f[t,i] = np.max(u)
                alpha[t,i] = np.argmax(u)
    zStar[n-1] = np.argmax(np.asarray(f[n-1, :]))
    for i in range(n-2, -1, -1):
        zStar[i] = alpha[i+1, int(zStar[i+1])]
    return zStar

def first_order(n, m, k, x, tol):
    #randomly initialize pi, phi and T#
    vals = np.random.rand(m)
    pi = np.log(vals/np.sum(vals))
    Tmat = np.zeros(shape = (m, m))
    phi = np.zeros(shape = (m, k))
    gamma = np.zeros(shape = (n, m))
    beta = np.zeros(shape = (n,m,m))
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    criteria = 0
    #cdef double[:,:] p = np.zeros(shape = (n,m))
    
    vals1 = np.random.rand(m,m)
    vals2 = np.random.rand(m,k)
    Tmat = np.log(vals1/np.sum(vals1, axis=1)[:,None])
    phi = np.log(vals2/np.sum(vals2, axis = 1)[:,None])
    
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        g = forwardAlg(n, m, k, pi, Tmat, phi, x)
        h = backwardAlg(n, m, k, pi, Tmat, phi, x)
        pNew = pForward(g, x)
        
        ##E-Step##
    
        #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        #p = np.full((n,m), pNew)
        #gamma = g+h-p
        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    beta[t,i,j] = Tmat[i,j] + phi[j, x[t]] + g[t-1, i] + h[t, j] - pNew
        ##M-Step##
    
        #Update pi, phi and Tmat#
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                Tmat[i,j] = logSumExp(beta[1::, i, j]) - logSumExp(beta[1::, i,:])
        for i in range(0,m):
            for w in range(0, k):
                j = 0
                count = 0
                for t in range(0,n):
                    if x[t] == w:
                        count = count+1
                indicies = np.zeros(count)
                for t in range(0,n):
                    if x[t] == w:
                        indicies[j] = gamma[t,i]
                        j = j+1
                    
                phi[i,w] = logSumExp(indicies) - logSumExp(gamma[:,i])
        
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
            print(iterations)
    return (iterations, pNew, np.exp(pi), np.exp(phi), np.exp(Tmat))
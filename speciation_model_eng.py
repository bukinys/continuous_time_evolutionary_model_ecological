#connecting required libraries
from itertools import product
import numpy as np
import time
import pandas as pd

#function to generate a table (matrix) of possible system states, states are encoded by a binary vector. The dimension of the vector equals the number of ecological niches.
#the vector value equals 1 - the ecological niche is occupied, 0 - the ecological niche is free
#the vectors are combined into an array, the array dimension equals the number of possible system states
#input parameter fn_niches - the number of ecological niches in the system
def state_matrix(fn_niches):
  c_matrix=[list(comb) for comb in product([0, 1], repeat=fn_niches)] #generating an array of system state vectors
  c_matrix=np.array(c_matrix)
  c_matrix=np.delete(c_matrix, 0, axis=0) #removing the vector with zero values - the state when species are absent in the system
  return c_matrix

#function to display a table (matrix) of possible system states on the screen
#parameter fmatrix - array with vectors encoding system states
def print_matrix(fmatrix):
  for row in fmatrix:
       print(row)


#function to display information on the number of states with n_occupied (n occupied) ecological niches
#parameter fmatrix - array with vectors encoding system states
#input parameter fn_niches - the number of ecological niches in the system
def print_niches(fmatrix, fn_niches):
  row_sums = [sum(row) for row in fmatrix] #calculating the number of occupied ecological niches in each state
  for i in range(0, fn_niches+1):
    number_occupied_ecological_niches=i
    number_of_states = row_sums.count(number_occupied_ecological_niches)
    print("\number of states with ", number_occupied_ecological_niches, " occupied niches => ", number_of_states)

#function to assess the possibility of transition between states in the system, if the transition is possible, the function returns 1, if not possible, returns 0
#parameter fmatrix - array with vectors encoding system states
#parameter fR_sos - numeric vector containing information about the number of occupied ecological niches in each state
#parameter n1 - number of the state from which the possibility of transition is assessed
#parameter n2 - number of the state to which the possibility of transition is assessed
def prob_transition(fmatrix, fR_sos, n1, n2):
  mismatches = np.sum(fmatrix[n1] != fmatrix[n2])
  if(mismatches==1 and fR_sos[n1]<fR_sos[n2]):
    return 1
  return 0

#function to generate a square matrix of allowed transitions between states, value 0 in the matrix means transition between states i and j is forbidden, value 1 - transition is allowed
#rows and columns in the matrix - state numbers
#parameter fmatrix - array with vectors encoding system states
#parameter fR_sos - numeric vector containing information about the number of occupied ecological niches in each state
def pb_transition(fmatrix, fR_sos):
  ns=len(fmatrix)
  pp_matrix = np.zeros((ns, ns), dtype=int)
  for i in range(0, ns):
    for j in range(0, ns):
      pp_matrix[i,j]=prob_transition(fmatrix, fR_sos, i, j)
  return pp_matrix #output matrix of allowed transitions

#function to assess the instantaneous probability of transition into state k from other states
#parameter fpp_matrix - matrix of allowed transitions
#parameter fPs - vector of current probabilities of the system being in each of the states 
def poz_cum_prob_transition(fpp_matrix, fPs, k):
  mask = (fpp_matrix[:, k] == 1)  # truth mask
  mask[k] = False                 # exclude k
  return np.sum(fPs[mask])        # sum probabilities by mask

#function to assess the instantaneous probability of transition from state k to other states
#parameter fpp_matrix - matrix of allowed transitions
#parameter fPs - vector of current probabilities of the system being in each of the states 
def neg_cum_prob_transition(fpp_matrix, fPs, k):
  mask = (fpp_matrix[k, :] == 1)  # truth mask
  mask[k] = False                 # exclude k
  return np.sum(mask)*fPs[k]      # sum probabilities by mask

#function for integration using the Euler method with recalculation with estimation of the probability of the absorbing state over time
#input parameter n_niches - the number of ecological niches in the system
#input parameter Lp - probability of species divergence
#input parameter dt - value of the time step for the Euler method with recalculation
def integrate_eil_per(n_niches, Lp, dt):

    #start of the integration algorithm
    st_matrix = state_matrix(n_niches) #generate the matrix of system states
    n_state=len(st_matrix) #number of states in the system
    print(f"\ntotal system states=> {len(st_matrix)}")
    R_sos = np.array([sum(row) for row in st_matrix]) #vector of the number of occupied niches in a state
    print_niches(st_matrix, n_niches) #display information on the number of states corresponding to occupied niches

    pp_matrix=pb_transition(st_matrix, R_sos) #generate the matrix of possible transitions
    print(f"\ndimension of the matrix of allowed transitions=> {len(pp_matrix)}")   


    Ps=np.zeros(n_state) #initialize the array of state probabilities
    Psp=np.zeros(n_state) #initialize the array of state probabilities for the intermediate array in the Euler integration algorithm with recalculation
    Psp2=np.zeros(n_state) #initialize the array of state probabilities for the intermediate array in the integration
    f1=np.zeros(n_state) #initialize the array variable for storing the intermediate value of the derivative (right-hand sides of the equation) for the Euler method with recalculation

    Ps[0]=1 #setting the initial condition - the probability of one of the states with one species equals 1


    #creating a dataframe table to store the step time, the probability of the system being in states with num occupied niches and the probability of the absorbing state
    df_pvale = pd.DataFrame(columns=range(1+n_niches)).astype(float) #table for storing the step time - first column and the probability of the system being in states with num occupied niches and the probability of the absorbing state
    col_num=np.array(range(1, n_niches+1)) #column names - identifiers of information about the number of occupied niches
    col_num="num_"+col_num.astype(str)
    df_pvale.columns = np.append("time", col_num)  #column names - identifier of the time column and information about the number of occupied niches
    df_pvale.loc[len(df_pvale)] = np.zeros(n_niches+1)
    df_pvale.iloc[0,1]=1 #adding information that at the initial time moment 0 the system had one species with probability 1

    t_start=int(time.time()) #start time of the computation in seconds for performance control

    time_step=0 #setting the starting time in the integrated system
    while Ps[n_state-1] <=0.99: #integration stopping rule, integration stops if the probability of the absorbing state <=0.99

      time_step+=dt #increasing the current time by step dt

      #first action for the Euler method with recalculation, computing the intermediate value of the derivative and the intermediate value of the function
      for j in range(0, len(Ps)):
        poz=poz_cum_prob_transition(pp_matrix, Ps, j)
        neg=neg_cum_prob_transition(pp_matrix, Ps, j)
        f1[j]=Lp*(poz-neg) #intermediate value of the derivative
        Psp2[j]=Ps[j]+f1[j]*dt #intermediate value of the function

      #second action for the Euler method with recalculation, computing the final value of the function
      for j in range(0, len(Ps)):
        poz=poz_cum_prob_transition(pp_matrix, Psp2, j)
        neg=neg_cum_prob_transition(pp_matrix, Psp2, j)
        Psp[j]=Ps[j]+(f1[j]+Lp*(poz-neg))*(dt/2) #final value of the function based on the averaged derivative value over the integration step

      #updating values
      Ps=Psp.copy()

      #adding data to the resulting dataframe
      pv_vector=np.zeros(n_niches+1)
      pv_vector[0]=np.round(time_step, decimals=3)

      #calculating the probabilities of the system being in states with j occupied ecological niches
      for j in range(1, n_niches+1):
        indices = np.where(R_sos == j) #indices of state elements in which j niches are occupied
        pv_vector[j]=sum(Ps[indices])  #extracting the probabilities of states in which j niches are occupied and summing these probabilities 
      df_pvale.loc[len(df_pvale)] = pv_vector

      #displaying the current probability of the absorbing state on the screen
      if len(df_pvale) % 100 == 0:
        print("\n", "num_niches=>", n_niches, " time=> ", np.round(time_step, decimals=3), " p_value=> ", np.round(Ps[n_state-1], decimals=4))

    t_final=int(time.time()) #end time of the computation in seconds
    print(f"\ncomputation time=> {t_final-t_start}\n") #displaying the time spent on the computation on the screen

    return df_pvale #output result - dataframe table with the step time, the probability of the system being in states with num occupied niches and the probability of the absorbing state

#start of the algorithm


path="path_to_directories_for_saving_calculation_results" #path to the directory for saving the result

Lpm=np.arange(0.01, 0.21, 0.01) #setting the vector of values of the probabilities of occupying a new niche (probabilities of species divergence) per unit time
dt=0.1 #setting the integration step for the Euler method with recalculation

#launching the algorithm for studying the system with analysis for all combinations of transition probabilities Lpm and the number of ecological niches varying in the range from 2 to 15
for i in range(0, len(Lpm)): #outer loop over probabilities
  for j in range(2, 16): #inner loop over the number of ecological niches
    n_niches=j #setting the value for the number of ecological niches parameter
    Lp=Lpm[i] #setting the value for the species divergence probability parameter
    df_pvalue=integrate_eil_per(n_niches, Lp, dt) #launching the integration
    st_file=f"{path}rez_Lp-{np.round(Lp, decimals=2)}_Nniches-{n_niches}.tsv" 
    df_pvalue.to_csv(st_file, sep='\t', index=False, encoding='utf-8') #saving the table with the step time and probabilities to disk in the path directory

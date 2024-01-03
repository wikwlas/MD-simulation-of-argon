"""
Authors: Viktoriia Vlasenko
Index number: 317013
"""

#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import time as simulation_time

#CONSTANTS
k = 0.00831


#function for saving positions of the atoms
def save_file(file, positions):
    with open(file, 'w') as file:
        file.write(f'{N}\n')
        file.write(f'argon\n')
        for i, (x,y,z) in enumerate(positions):
            file.write(f'Ar\t{x}\t{y}\t{z}\n')


if __name__ == '__main__':
    #Parameters list
    parameters = []
    file_with_parameters = "parameters.txt"


    with open(file_with_parameters) as file_with_parameters:
            for line in file_with_parameters.readlines():
                parameters.append(float(line.split()[0]))

    print("Received parameters from the file: ", parameters)
    n, m, epsilon, R, f, a, T_0, tau, S_o, S_d, S_out, S_xyz = parameters
    n=int(n)
    S_o=int(S_o)
    S_d=int(S_d)
    S_out=int(S_out)
    S_xyz=int(S_xyz)


    file_with_positions= open(f"sym n={n} T={T_0}.txt", 'w')
    file_with_energies = open("total_energies.txt", 'w')


    N=n**3
    L = 0.5*a*(n-1)*np.sqrt(6)
    

    b_0=np.array([a, 0, 0])
    b_1=np.array([a/2, (a/2)*np.sqrt(3), 0])
    b_2=np.array([a/2, (a/6)*np.sqrt(3), a*np.sqrt(2/3)])

    #positions and moments of the atoms
    r = np.zeros((N, 3))
    p = np.zeros((N, 3))


    for i_0 in range(n):
         for i_1 in range(n):
              for i_2 in range(n):
                   i = i_0 + i_1*n + i_2*(n**2)
                   r[i, :] = (i_0 - (n-1)/2)*b_0 + (i_1 - (n-1)/2)*b_1 + (i_2 - (n-1)/2)*b_2
                   #p[i, :] = np.sqrt(2*m*E_kin[i, 0]), np.sqrt(2*m*E_kin[i, 1]), np.sqrt(2*m*E_kin[i, 2])
    
    E_kin=np.array(-0.5*k*T_0*np.log(np.random.random(size=(N, 3))))

    for i, (x, y, z) in enumerate(np.random.randint(2, size=(N, 3)) * 2 - 1):
        p[i, :] = x*np.sqrt(2*m*E_kin[i, 0]), y*np.sqrt(2*m*E_kin[i, 1]), z*np.sqrt(2*m*E_kin[i, 2])

    p = p - np.sum(p)/N 
    
    save_file('data.xyz', r)


    # #Histograms
    # fig, ax = plt.subplots(2, 2, dpi=150, figsize=(8, 8))
    # ax[0, 0].hist(p[:, 0], bins=50)
    # ax[0, 0].set_title('Axis X')
    # ax[0, 1].hist(p[:, 1], bins=50)
    # ax[0, 1].set_title('Axis Y')
    # ax[1, 0].hist(p[:, 2], bins=50)
    # ax[1, 0].set_title('Axis Z')
    # plt.show()


    global V, H, F, P, T, t, H_s, V_s, V_p, F_s, F_p
    t = 0
    H_s = 0
    V_s = np.zeros(N)
    V_p = np.zeros(N)
    F_s = np.zeros((N,3))
    F_p = np.zeros((N,3))

    @jit(nopython=True)
    def potential_force(r):
        Vp = np.zeros(N)
        Vs = np.zeros(N)
        Fp = np.zeros((N, 3))
        Fs = np.zeros((N, 3))

        for i in range(N):
            r_abs1 = np.linalg.norm(r[i])
            if r_abs1 >= L:
                x = r_abs1 - L
                Vs[i] = (f * x**2)/2
                Fs[i] = -f * x * r[i] / r_abs1
            for j in range(i):
                r_abs2 = np.linalg.norm(r[i] - r[j])
                y = (R/r_abs2)**6
                Vp[i] += epsilon*y*(y - 2)
                diff = 12*epsilon*y*(y - 1)*(r[i] - r[j])/r_abs2**2
                Fp[i] += diff
                Fp[j] -= diff
        V = np.sum(Vp) + np.sum(Vs)
        F = Fs + Fp
        return V, Fs, Fp, F

    def energy(p):
        norm = np.linalg.norm(p,axis=-1)
        Ek = ((1/(2*m))*norm**2)
        return Ek
    
    def total_e(p,V):
        norm = np.linalg.norm(p,axis=-1)
        H = (np.sum(norm**2/(2*m))+V)
        return H

    def temperature(E):
        T=2/(3*N*k)*np.sum(E)
        return T
    
    def pressure(F_s):
        P = (np.sum(np.linalg.norm(F_s,axis=-1))/(4*np.pi*L**2))
        return P
    
    def save_energies(H,V,T,P,t,p):
        file_with_energies.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(t, H, np.sum(energy(p)), V, T, P))

    def save_positions(r, E_kin, N):
        # with open(file_with_positions, 'w') as file:
        file_with_positions.write(f'{N}\n\n')
        for (x, y, z), E in zip(r, E_kin):
            file_with_positions.write(f'Ar\t{x}\t{y}\t{z}\n')


    #Simulation
    def motion_simulation(r, p, tau, F, V, P, t, T, V_s, V_p, F_s, F_p, S_o, S_d, S_xyz, S_out):
        H_s = 0
        
        for i in range(S_o + S_d):
            p += (1/2)*F*tau
            r += (1/m)*p*tau
            V, F_s, F_p, F = potential_force(r)
            P = pressure(F_s)
            p += (1/2)*F*tau
            t += tau

            E_kin = energy(p)
            H = total_e(p,V)
            T = temperature(E_kin)
              
            if not (i % S_out):
                save_energies(H,V,T,P,t,p)

            if not (i % S_xyz):
                save_positions(r,E_kin,N)

            if (i >= S_o):    
                H_s += H
    
        H_s /=S_d
        return H_s


    V, F_s, F_p, F = potential_force(r)
    P = None
    T = None

    start = simulation_time.time()
    H_s = motion_simulation(r, p, tau, F, V, P, t, T, V_s, V_p, F_s, F_p, S_o, S_d, S_xyz, S_out)
    stop = simulation_time.time()
    print(f'Hs = {H_s}')
    print(f'Simulation time for n = {n} is {stop - start} s, {(stop - start)/60} min')
    file_with_positions.close()
    file_with_energies.close()


    




    


        



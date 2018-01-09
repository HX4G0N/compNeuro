#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:57:19 2017


A collection of neuron models, implemented in numpy
Calls to all models are identical and require a set of timesteps (which do not necessary
need to be equally spaced!) and a matrix or vector holding input currents.
It is possible to simulate multiple neurons in parallel, in this case simply pass a matrix
shaped ``nb_timesteps, nb_neurons`` as the input current.
By default, the first timestep is initialized by the steady state of the model assuming
a current ``I[0,:] = 0``, so it might benefit the simulation to have the first non-zero
entries at the second timestep or later.


@author: carlodedonno
"""

import numpy as np

def hodgkin_huxley(tt, I, u_rest=-65.):
    """ Simulation of Hodkin-Huxley neuron model

    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    """

    #SODIUM CHANNEL (m, h are the gating variables)
    alpha_m = lambda u: (2.5 - 0.1 * (u - u_rest)) / (np.exp(2.5 - 0.1 * (u - u_rest)) - 1)
    alpha_h = lambda u: (0.07 * np.exp((-u + u_rest) / 20))
    beta_m = lambda u: 4 * np.exp((-u + u_rest) / 18)
    beta_h = lambda u: 1 / (np.exp(3 - 0.1 * (u - u_rest)) + 1)
    g_Na = 120
    E_Na = 55

    #POTASSIUM (n is the gating variable)
    alpha_n = lambda u: (0.1 - 0.01 * (u - u_rest)) / (np.exp(1 - 0.1 * (u - u_rest)) - 1)
    beta_n = lambda u: 0.125 * np.exp((-u + u_rest) / 80)
    g_K = 36
    E_K = -75

    #other channels
    g_L = 0.3
    E_L = -69

    #capacitance
    C = 0.1

    #initialize gating variables and voltage
    h, m, n, u = [np.zeros_like(I) for i in range(4)]
    h[0] = alpha_h(u_rest) / (alpha_h(u_rest) + beta_h(u_rest))
    m[0] = alpha_m(u_rest) / (alpha_m(u_rest) + beta_m(u_rest))
    n[0] = alpha_n(u_rest) / (alpha_n(u_rest) + beta_n(u_rest))
    u[0] = u_rest

    #calculate using explicit euler method
    for t in range(1, len(tt)):
        dt = tt[t] - tt[t-1]

        dhdt = alpha_h(u[t - 1]) * (1 - h[t - 1]) - beta_h(u[t - 1]) * h[t - 1]
        dmdt = alpha_m(u[t - 1]) * (1 - m[t - 1]) - beta_m(u[t - 1]) * m[t - 1]
        dndt = alpha_n(u[t - 1]) * (1 - n[t - 1]) - beta_n(u[t - 1]) * n[t - 1]
        dudt = ( g_Na * (E_Na - u[t-1]) * m[t-1]**3 * h[t-1] \
               + g_K  * (E_K - u[t-1]) *  n[t-1]**4 \
               + g_L  * (E_L - u[t-1]) + I[t-1]) / C

        h[t] = h[t - 1] + dt * dhdt
        m[t] = m[t - 1] + dt * dmdt
        n[t] = n[t - 1] + dt * dndt
        u[t] = u[t - 1] + dt * dudt

    return h, m, n, u

def hodgkin_huxley_noise(tt, I, u_rest=-65., mean=0, std=0.002):
    """ Simulation of Hodkin-Huxley neuron model

    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    """
    
    #SODIUM CHANNEL (m, h are the gating variables)
    alpha_m = lambda u: (2.5 - 0.1 * (u - u_rest)) / (np.exp(2.5 - 0.1 * (u - u_rest)) - 1)
    alpha_h = lambda u: (0.07 * np.exp((-u + u_rest) / 20))
    beta_m = lambda u: 4 * np.exp((-u + u_rest) / 18)
    beta_h = lambda u: 1 / (np.exp(3 - 0.1 * (u - u_rest)) + 1)
    g_Na = 120
    E_Na = 55
    noise_Na = np.random.normal(mean, std, I.shape)

    #POTASSIUM (n is the gating variable)
    alpha_n = lambda u: (0.1 - 0.01 * (u - u_rest)) / (np.exp(1 - 0.1 * (u - u_rest)) - 1)
    beta_n = lambda u: 0.125 * np.exp((-u + u_rest) / 80)
    g_K = 36
    E_K = -75
    noise_K = np.random.normal(mean, std, I.shape)
    #create noiseNa and noiseK vectors

    #other channels
    g_L = 0.3
    E_L = -69

    #capacitance
    C = 0.1

    #initialize gating variables and voltage
    h, m, n, u = [np.zeros_like(I) for i in range(4)]
    h[0] = alpha_h(u_rest) / (alpha_h(u_rest) + beta_h(u_rest))
    m[0] = alpha_m(u_rest) / (alpha_m(u_rest) + beta_m(u_rest))
    n[0] = alpha_n(u_rest) / (alpha_n(u_rest) + beta_n(u_rest))
    u[0] = u_rest

    #calculate using explicit euler method
    for t in range(1, len(tt)):
        dt = tt[t] - tt[t-1]

        dhdt = alpha_h(u[t - 1]) * (1 - h[t - 1]) - beta_h(u[t - 1]) * h[t - 1]
        dmdt = alpha_m(u[t - 1]) * (1 - m[t - 1]) - beta_m(u[t - 1]) * m[t - 1]
        dndt = alpha_n(u[t - 1]) * (1 - n[t - 1]) - beta_n(u[t - 1]) * n[t - 1]
        dudt = ( g_Na * (E_Na - u[t-1]) * (m[t-1]**3 * h[t-1] + noise_Na[t-1]) \
               + g_K  * (E_K - u[t-1]) *  (n[t-1]**4 + noise_K[t-1]) \
               + g_L  * (E_L - u[t-1]) + I[t-1]) / C

        h[t] = h[t - 1] + dt * dhdt
        m[t] = m[t - 1] + dt * dmdt
        n[t] = n[t - 1] + dt * dndt
        u[t] = u[t - 1] + dt * dudt

    return h, m, n, u

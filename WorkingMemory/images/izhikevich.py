from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from random import random
import scipy
import os
import imageio
# pip install torchdiffeq
from torchdiffeq import odeint
# pip install opencv-python

import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch.distributions import categorical as tdistc
from torch.distributions import bernoulli as tdistb
import gc

def model_parameters():

    # Initialize the params dictionary
    params = {}

    ## Timeline
    params['t_end'] = 2
    params['step'] =  0.0001;
    params['n'] = int(params['t_end'] / params['step'])

    ## Experiment
    params['learn_start_time'] = 0.5
    params['learn_impulse_duration'] = 0.2
    params['learn_impulse_shift'] = 0.3
    params['learn_order'] = [0]
    # params['learn_order'] = [x + 1 for x in params['learn_order']]  # Adjust for 1-based indices

    params['test_start_time'] = 0.8
    params['test_impulse_duration'] = 0.15
    params['test_impulse_shift'] = 0.4
    params['test_order'] = [0, 1]
    # params['test_order'] = [x + 1 for x in params['test_order']]  # Adjust for 1-based indices

    ## Applied pattern current
    params['variance_learn'] = 0.05
    params['variance_test'] = 0.2
    params['Iapp_learn'] = 80
    params['Iapp_test'] = 8

    ## Movie
    params['after_sample_frames'] = 200
    params['before_sample_frames'] = 1

    ## Poisson noise
    params['poisson_nu'] = 1.5
    params['poisson_n_impulses'] = 15
    params['poisson_impulse_duration'] = int(0.03 / params['step'])
    params['poisson_impulse_initphase'] = int(1.5 / params['step'])
    params['poisson_amplitude'] = 20

    ## Runge-Kutta steps
    params['u2'] = params['step'] / 2
    params['u6'] = params['step'] / 6

    # ## Network size
    # params['mneuro'] = 79
    # params['nneuro'] = 79
    # params['quantity_neurons'] = params['mneuro'] * params['nneuro']
    params['mastro'] = 26
    params['nastro'] = 26
    az = 4  # Astrosyte zone size
    params['az'] = az - 1

    ## Initial conditions
    params['v_0'] = -70
    params['ca_0'] = 0.072495
    params['h_0'] = 0.886314
    params['ip3_0'] = 0.820204

    ## Neuron mode
    params['aa'] = 0.1  # FS
    params['a'] =0.1
    params['b'] = 0.2
    params['c'] = -65
    params['d'] = 2
    params['alf'] = 10
    params['k'] = 600
    params['neuron_fired_thr'] = 30
    params['I_input_thr'] = 25

    ## Synaptic connections
    params['N_connections'] = 4  # maximum number of connections between neurons
    # params['quantity_connections'] = params['quantity_neurons'] * params['N_connections']
    params['lambda'] = 5  # average exponential distribution
    params['beta'] = 5
    params['gsyn'] = 0.025
    params['aep'] = 0.5  # astrocyte effect parameter
    params['Esyn'] = 0
    params['ksyn'] = 0.2

    ## Astrosyte model
    params['dCa'] = 0.05
    params['dIP3'] = 0.1  # 0.05
    params['enter_astro'] = 6
    params['min_neurons_activity'] = 8
    params['t_neuro'] = 0.06
    params['amplitude_neuro'] = 5
    params['threshold_Ca'] = 0.15
    window_astro_watch = 0.01  # t(sec)x.
    shift_window_astro_watch = 0.001  # t(sec)
    impact_astro = 0.25  # t(sec)
    params['impact_astro'] = int(impact_astro / params['step'])
    params['window_astro_watch'] = int(window_astro_watch / params['step'])
    params['shift_window_astro_watch'] = int(shift_window_astro_watch / params['step'])

    ## Memory performance
    params['max_spikes_thr'] = 30
    return params
params = model_parameters()
### Standard
class GraphemeColourSynaesthesiaNet(nn.Module):
  def __init__(self, input_dim, M, tau=1.0,tolfun=4e-005,  eta=0.01, eta_w =0.01, cross_talk = True, modalities = 2, FF = False):
    super(GraphemeColourSynaesthesiaNet, self).__init__()

    self.modalities = input_dim[0] # Amount of modalities
    self.cross_talk = cross_talk # True or False
    self.FF = FF
    # Neurons
    self.tau =  nn.Parameter(torch.tensor(tau)) # time constant
    self.eta_k = nn.Parameter(torch.tensor(eta))
    self.eta_w = nn.Parameter(torch.tensor(eta_w))
    self.input_dim = input_dim[-1]
    self.N = self.input_dim*2
    self.M = M
    self.g_p_1 = 0
    self.g_p_2 = 0
    self.time_step = self.dt = 0.01 # steps taken in ms
    # self.transferFunction = 2 # 1 = tanh,  2= logistic function. liberman favourite parameter.

    self.critical_eta = None # critical learning rate
    self.variance = None # output variance for modalities
    self.tolfun = tolfun

    self.g = nn.Sigmoid()
    # debatable whether M or modalities
    if cross_talk:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=True) # start at 0 fixed point or near 0

      # Generate random angles
      theta = 2 * np.pi * torch.rand(self.M**2)

      # Set radius for the ring
      radius = 0.1  # Can be adjusted

      # Convert to cartesian coordinates
      K12 = radius * torch.cos(theta)
      with torch.no_grad():
        self.K = nn.Parameter(torch.reshape(K12, (self.M, self.M)))
            # self.K[j][i]= K21
      with torch.no_grad():
        self.K.fill_diagonal_(0) # no self connections
    else:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=False) # start at 0 fixed point
    # Initialise feedforward weights
    if self.FF:
      self.W = nn.Parameter(torch.randn(self.M, self.N), requires_grad=True)
    else:
      self.W = nn.Parameter(torch.randn(self.M, self.N), requires_grad=False)

    self.s1 = torch.zeros(self.M//2, requires_grad=False) # Membrane potential or output neurons
    self.s2 = torch.zeros(self.M//2, requires_grad=False) # Membrane potential or output neurons
    self.optimizer = torch.optim.Adam(self.parameters(), lr=eta)
    self.losses =  None
    self.variances = None
    self.critical_etas = None
    assert self.N <= self.M, 'This is an overcomplete network use more output than input neurons'
  def steady_state(self, x, s1, s2):
    '''
    The steady state interactions
    '''
    # Apply first order activities to the output perception s
    s = torch.stack([self.s1, self.s2]).flatten()
    s = self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s))
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    return self.s1, self.s2

  def feedforward_dynamics(self, x, t, s1, s2, samp_ts):
    '''
    The feedforward dynamics  of the model
    note: original math and paper do not consider modalities
    '''
    s = torch.stack([s1, s2]).flatten()
    # try:
    # s = odeint(self.dynamics, s, samp_ts[:t+1])[-1]
    s = self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s))
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    # except:
      # self.s1, self.s2 = s1, s2
    return

  def objective_function(self, x, s1, s2):
      '''
      The cost function use for estimating the
      mutual information between the in and output neurons.
      Is used as a loss or cost function

      note: original math and paper do not consider modalities
      does not work with  k = 0
      '''
      s = torch.stack([s1, s2]).flatten()
      g_p_1 = self.logistic_derivative(torch.matmul(self.W, x) + torch.matmul(self.K, s))
      # g_p_2 = self.logistic_derivative(self.W2@x[1] + K21 * s1)
      G = torch.diag(g_p_1)
      # G_2 = torch.diag(g_p_2)
      if torch.any(self.K != 0):
        phi = ((G)**(-1) - self.K)**(-1)
        # phi_2 = ((G_2)**(-1) - K21)**(-1)
        chi = (phi @ self.W)
      else:
        phi = G
        # phi_2 = G_2
        chi = phi @ self.W
        # chi_2 = phi_2 * self.W2
        # print(np.shape(chi1))
      E = - 0.5 * torch.trace(torch.log(chi.T @ chi))
      # E2 = - 0.5 * torch.trace(torch.log(chi_2.T @ chi_2))
      return E, chi, G, phi

  def learning_rule(self, x, E):
    if self.FF:
      w_delta = - self.eta_w * torch.autograd.grad(E, self.W, retain_graph=True)[0]
      with torch.no_grad():
        self.W = nn.Parameter(self.W.clone() + w_delta)

    if self.cross_talk:
      k_delta = - self.eta_k * torch.autograd.grad(E, self.K, retain_graph=True)[0]

      with torch.no_grad():
        self.K += k_delta
        self.K.fill_diagonal_(0)
    return


  def stable(self, x, s1, s2):
    alpha1 = torch.mean(s1**2)
    alpha2 = torch.mean(s2**2)
    Tr = 4 * alpha1 * alpha2 - alpha1 - alpha2
    Det = - 9/16 +3*alpha1 +3*alpha2 - 4*alpha1**2 - 4 * alpha2**2 - (29/2)*alpha1*alpha2 +18*alpha1*alpha2**2 +18*alpha1**2*alpha2 - 21*alpha1**2*alpha2**2
    self.critical_eta = -(Tr + torch.sqrt(Tr**2 - 4 * Det))/Det
    gamma1 = abs(1+0.5*self.eta_k*(Tr + torch.sqrt(Tr**2-4*Det)))
    gamma2 = abs(1+0.5*self.eta_k*(Tr - torch.sqrt(Tr**2-4*Det)))
    if gamma1 < 1 and gamma2 < 1:
     return True
    return False
  #   return Det
  def convergence(self, s1_prev, s2_prev, s1, s2, loss, i):
    diff1 = abs(s1 - s1_prev)
    diff2 = abs(s2 - s2_prev)
    # difference  = torch.mean(torch.hstack([diff1, diff2])) < self.tolfun
    difference = abs(self.losses[i-1]- loss) < self.tolfun
    return difference.all()

  def forward(self, x, max_iter=10):
    status = []
    self.timeline = int((max_iter)/self.time_step)
    self.spikes= torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities, self.N ))
    self.losses = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    self.variances = torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities))
    self.critical_etas = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    x = x.flatten()
    self.x =x
    sample_ts = torch.linspace(0., int((max_iter)/self.time_step), int(max_iter/self.time_step))
    converged = []
    s1_prev, s2_prev = -np.inf, -np.inf


    for i in tqdm(range(1,int((max_iter/self.time_step)))):
     

      with torch.no_grad():
        s1 = self.s1.clone()
        s2 = self.s2.clone()


        if (abs(self.K) <= 1).all():
          status.append('Stable')
          s1, s2 = self.steady_state(x, s1, s2)
          spiked, fired = self.Izhikevich_neurons.step(s1, s2, self.time_step)
        else:
          status.append('Unstable')

      self.feedforward_dynamics(x, i, s1, s2, samp_ts)

      loss, chi, G, phi = self.objective_function(x, s1, s2)
      if self.convergence( s1_prev, s2_prev, s1, s2, loss, i): # steady-stae
        converged.append(i)
        self.FF = False
      self.learning_rule(x, loss)


      self.losses[i] = loss
      loss.backward()

      self.optimizer.step()
      self.optimizer.zero_grad()

      self.stable(x, s1, s2)

      self.variances[i,0] = torch.mean(self.s1**2) - torch.mean(self.s1)**2
      self.variances[i,1] = torch.mean(self.s2**2) - torch.mean(self.s2)**2
      self.critical_etas[i] = self.critical_eta
     # self.critical_etas[i] = self.critical_eta
      s1_prev, s2_prev = s1.clone(), s2.clone()
      with torch.no_grad():
        self.K.fill_diagonal_(0)
    # if len(converged) != 0:
    #   print('\n --- \n Converged at iteration: ', converged[0], '\n --- \n')
    #   # print('\n --- \n Diverged or reached max iterations after at iteration: ', converged[-1], '\n --- \n')
    # else:
    #   print('\n --- Did not converge \n ---\n ')
    return status, converged

  def predict(self, x, s):
    return self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s))

  #helper functions
  def dynamics(self, t, s):
    # print((self.W1 @ x[0] + self.K[0][1].clone() * s[1]))
    s_dt = -s + self.g(self.W @ self.x + self.K @ s)/self.tau
    s1_dt, s2_dt = s_dt[:self.M//2], s_dt[self.M//2:]
    return torch.stack([s1_dt, s2_dt]).flatten()
  def logistic_derivative(self,x):
    return self.g(x) * (1 - self.g(x))
  def second_logistic_derivative(self, x):
    return self.logistic_derivative(x) * (1 - 2*self.g(x))
  def third_logistic_derivative(self, x):
    return self.second_logistic_derivative(x)* (1 - 2*self.g(x)) - 2*self.logistic_derivative(x)**2

### Izhikevich
class IzhikevichNeuron():
  def __init__(self, params, M, N):
    super(IzhikevichNeuron, self).__init__()
    # Time grid
    random_factor = random()
    # if (neuron_type == 'excitatory' or 'excit'): all neurons should be excitatory according to wm and shriki background
    self.a = params['a']
    self.b = params['b']
    self.c = params['c'] + 15*random_factor**2
    self.d = params['d'] - 6*random_factor**2
    # elif (neuron_type == 'inhibitory' or 'inhib'):
    #     a = params['a'] + 0.08*random_factor
    #     b = params['b'] - 0.05*random_factor
    #     c = params['c']
    #     d = params['d']
    # else:
    #     return 'Neuron type must be excitatory or inhibitory'
    self.v = torch.full((N, M), self.a) # Membrane potential
    self.u = torch.full((N, M), self.b * self.c ) # Recovery variable
  def step(self, I1, I2, time_step):
    I = torch.stack([I1, I2]).flatten()
    # I = I.view(*self.v.shape)  # Reshape I to match v's dimensions
    # I = torch.repeat_interleave(I, self.v.size(dim=1), axis=0)
    # Find neurons that fired
    # Update fired neurons
    # fired = fired_mask = torch.where(self.v >= params['neuron_fired_thr'], 1, 0)

    n_substeps = 1
    dt = time_step / n_substeps
    I = torch.stack([I1, I2]).flatten()
    
    fired = torch.zeros_like(self.v, dtype=torch.int)
    
    # Break the timestep into smaller steps for stability
    for _ in range(n_substeps):
      # Identify neurons currently above threshold
      fired_now = self.v >= params['neuron_fired_thr']
      fired = torch.logical_or(fired, fired_now)  # Track any firing during the full timestep
      
      # Reset neurons that just fired
      self.v[fired_now] = self.c
      self.u[fired_now] = self.u[fired_now] + self.d
      
      # Compute updates for all neurons
      dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + I
      du = self.a * (self.b * self.v - self.u)
      
      # Apply updates only to neurons below threshold
      not_fired_now = ~fired_now
      self.v[not_fired_now] += dv[not_fired_now] * dt
      self.u[not_fired_now] += du[not_fired_now] * dt
            # # solve using RK 4th order

            # dv1 = time_step * dvdt(v_aux, u_aux, I_aux)
            # dv2 = time_step * dvdt(v_aux + dv1 * 0.5, uc, Ic)
            # dv3 = time_step * dvdt(v_aux + dv2 * 0.5, uc, Ic)
            # dv4 = time_step * dvdt(v_aux + dv3, u_aux, I_aux)
            # v[t] = 1/6 * (dv1 + 2*(dv2 + dv3) + dv4)

            # du1 = time_step * dudt(v_aux, u_aux)
            # du2 = time_step * dudt(v_aux, u_aux + du1 * 0.5)
            # du3 = time_step * dudt(v_aux, u_aux + du2 * 0.5)
            # du4 = time_step * dudt(v_aux, u_aux + du3)
            # u[t] = 1/6 * (du1 + 2*(du2 + du3) + du4)

    # return membrane potential and input current
    return self.v, fired
class GraphemeColourSynaesthesiaSpikeNet(nn.Module):
  def __init__(self, params, input_dim, M, tau=1.0,tolfun=4e-005,  eta=0.01, eta_w =0.01, cross_talk = True, modalities = 2, FF = True):
    super(GraphemeColourSynaesthesiaSpikeNet, self).__init__()

    self.modalities = input_dim[0] # Amount of modalities
    self.cross_talk = cross_talk # True or False
    self.FF = FF
    # Neurons
    self.tau =  nn.Parameter(torch.tensor(tau)) # time constant
    self.eta_k = nn.Parameter(torch.tensor(eta))
    self.eta_w = nn.Parameter(torch.tensor(eta_w))
    self.input_dim = input_dim[-1]
    self.N = self.input_dim*2
    self.M = M
    self.g_p_1 = 0
    self.g_p_2 = 0
    self.time_step = self.dt = 0.01 # steps taken in ms

    self.critical_eta = None # critical learning rate
    self.variance = None # output variance for modalities
    self.tolfun = tolfun

    self.g = nn.Sigmoid()
    self.Izhikevich_neurons = IzhikevichNeuron(params, self.M, self.N)
    # debatable whether M or modalities
    if cross_talk:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=True) # start at 0 fixed point or near 0

      # Generate random angles
      theta = 2 * np.pi * torch.rand(self.M**2)

      # Set radius for the ring
      radius = 0.1  # Can be adjusted

      # Convert to cartesian coordinates
      K12 = radius * torch.cos(theta)
      with torch.no_grad():
        self.K = nn.Parameter(torch.reshape(K12, (self.M, self.M)))
      with torch.no_grad():
        self.K.fill_diagonal_(0) # no self connections
    else:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=False) # start at 0 fixed point
    # Initialise feedforward weights
    if self.FF:
      self.W = nn.Parameter(torch.randn(self.M, self.N), requires_grad=True)
    else:
      self.W = nn.Parameter(torch.randn(self.M, self.N), requires_grad=False)

    self.s1 = torch.zeros(self.M//2, requires_grad=False) # Membrane current or output neurons
    self.s2 = torch.zeros(self.M//2, requires_grad=False) # Membrane current or output neurons
    self.optimizer = torch.optim.Adam(self.parameters(), lr=eta)
    self.spikes= None
    self.losses =  None
    self.variances = None
    self.critical_etas = None
    assert self.N <= self.M, 'This is an overcomplete network use more output than input neurons'
  def steady_state(self, x, s1, s2):
    '''
    The steady state interactions
    '''
    # Apply first order activities to the output perception s
    s = torch.stack([self.s1, self.s2]).flatten()
    s = self.g(self.W @ x + self.K @ s)
    s = s/max(s) * 1.5
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    return self.s1.clone(), self.s2.clone()

  def feedforward_dynamics(self, x, t, s1, s2, samp_ts):
    '''
    The feedforward dynamics  of the model
    note: original math and paper do not consider modalities
    '''
    s = torch.stack([s1, s2]).flatten()
    # try:
    # s = odeint(self.dynamics, s, samp_ts[:t+1])[-1]
    s = self.g(self.W @ x + self.K @ s)
    s = s/max(s) * 1.5
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    # except:
      # self.s1, self.s2 = s1, s2
    self.s1, self.s2 =  torch.clip(torch.stack([self.s1, self.s2 ]), 0.01, 1.5) # nA of sensory regions
    spikes, fired = self.Izhikevich_neurons.step(self.s1.clone(), self.s2.clone(), self.time_step)
    return spikes, fired


  def objective_function(self, x, s1, s2):
      '''
      The cost function use for estimating the
      mutual information between the in and output neurons.
      Is used as a loss or cost function

      note: original math and paper do not consider modalities
      does not work with  k = 0
      '''
      s = torch.stack([s1, s2]).flatten()
      g_p_1 = self.logistic_derivative(self.W@x+ self.K @ s)
      G = torch.diag(g_p_1)
      if torch.any(self.K != 0):
        phi = ((G)**(-1) - self.K)**(-1)
        chi = (phi @ self.W)
      else:
        phi = G
        chi = phi @ self.W
      E = - 0.5 * torch.trace(torch.log(chi.T @ chi))
      return E, chi, G, phi

  def learning_rule(self, x, E):
    if self.FF:
      w_delta = - self.eta_w * torch.autograd.grad(E, self.W, retain_graph=True)[0]
      with torch.no_grad():
        self.W = nn.Parameter(self.W.clone() + w_delta)

    if self.cross_talk:
      k_delta = - self.eta_k * torch.autograd.grad(E, self.K, retain_graph=True)[0]

      with torch.no_grad():
        self.K += k_delta
        self.K.fill_diagonal_(0)
    return


  def stable(self, x, s1, s2):
    alpha1 = torch.mean(s1**2)
    alpha2 = torch.mean(s2**2)
    Tr = 4 * alpha1 * alpha2 - alpha1 - alpha2
    Det = - 9/16 +3*alpha1 +3*alpha2 - 4*alpha1**2 - 4 * alpha2**2 - (29/2)*alpha1*alpha2 +18*alpha1*alpha2**2 +18*alpha1**2*alpha2 - 21*alpha1**2*alpha2**2
    self.critical_eta = -(Tr + torch.sqrt(Tr**2 - 4 * Det))/Det
    gamma1 = abs(1+0.5*self.eta_k*(Tr + torch.sqrt(Tr**2-4*Det)))
    gamma2 = abs(1+0.5*self.eta_k*(Tr - torch.sqrt(Tr**2-4*Det)))
    if gamma1 < 1 and gamma2 < 1:
     return True
    return False

  def convergence(self, s1_prev, s2_prev, s1, s2, loss, i):
    diff1 = abs(s1 - s1_prev)
    diff2 = abs(s2 - s2_prev)
    difference  = torch.mean(torch.hstack([diff1, diff2])) < self.tolfun
    difference = abs(self.losses[i-1]- loss) < self.tolfun
    return difference.all()

  def forward(self, x, max_iter=10):
    status = []
    self.timeline = int((max_iter)/self.time_step)
    self.spikes= torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.M, self.N ))
    self.fires= torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.M, self.N ))
    self.losses = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    self.variances = torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities))
    self.critical_etas = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    x = x.flatten()
    self.x =x
    samp_ts = torch.linspace(0., int((max_iter)/self.time_step), int(max_iter/self.time_step))
    converged = []

    s1_prev, s2_prev = -np.inf, -np.inf


    for i in tqdm(range(1,int((max_iter/self.time_step)))):
      # if i == 100:
      #   self.FF = False

      with torch.no_grad():
        s1 = self.s1.clone()
        s2 = self.s2.clone()


        if (abs(self.K) <= 1).all():
          status.append('Stable')
          s1, s2 = self.steady_state(x, s1, s2)
          s1, s2 =  torch.clip(torch.stack([s1, s2 ]),0.01, 1.5)

          spiked, fired = self.Izhikevich_neurons.step(s1, s2, self.time_step)
        else:
          status.append('Unstable')
      spiked, fired = self.feedforward_dynamics(x, i, s1, s2, samp_ts)

      loss, chi, G, phi = self.objective_function(x, s1, s2)
      if self.convergence( s1_prev, s2_prev, s1, s2, loss, i): # steady-stae
        converged.append(i)
        self.FF=False
      self.learning_rule(x, loss)


      self.losses[i] = loss
      loss.backward()

      self.fires[i] = fired
      self.spikes[i] = spiked

      self.optimizer.step()
      self.optimizer.zero_grad()

      # self.stable(x, s1, s2)

      self.variances[i,0] = torch.mean(self.s1**2) - torch.mean(self.s1)**2
      self.variances[i,1] = torch.mean(self.s2**2) - torch.mean(self.s2)**2
      # self.critical_etas[i] = self.critical_eta
      s1_prev, s2_prev = s1.clone(), s2.clone()
      with torch.no_grad():
        self.K.fill_diagonal_(0)
    if len(converged) != 0:
      print('\n --- \n Converged at iteration: ', converged[0], '\n --- \n')
      # print('\n --- \n Diverged or reached max iterations after at iteration: ', converged[-1], '\n --- \n')
    else:
      print('\n --- Did not converge \n ---\n ')
    return status, converged

  def predict(self, x, s):
    return self.g(self.W @ self.x + self.K @ s)

  #helper functions
  def dynamics(self, t, s):
    '''Time dynamics function describes regular updating and no retention of s'''
    s_dt = -s + self.g(self.W @ self.x + self.K @ s)/self.tau
    s1_dt, s2_dt = s_dt[:self.M//2], s_dt[self.M//2:]
    return torch.stack([s1_dt, s2_dt]).flatten()
  def logistic_derivative(self,x):
    return self.g(x) * (1 - self.g(x))
  def second_logistic_derivative(self, x):
    return self.logistic_derivative(x) * (1 - 2*self.g(x))
  def third_logistic_derivative(self, x):
    return self.second_logistic_derivative(x)* (1 - 2*self.g(x)) - 2*self.logistic_derivative(x)**2
### Simulation
# flat gray scale number array

def train(Izhikevich=True):
  img = cv2.imread('zero.jpg')
  img = cv2.resize(img, (0,0), fx=0.06, fy=0.06)

  x1 = torch.from_numpy((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()<127).astype('float32'))
  # colour category per pixel 0=original  blue=green
  x2 = torch.from_numpy(np.random.choice([0,1,2, 3], size = len(x1)).astype('float32'))
  x2[x1==0] = 0
  simulation_emergence_data = torch.stack([x1, x2])
  print(np.shape(simulation_emergence_data))
  E_Network = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)
  #syn
  weights1 = E_Network.W
  emergence_iterations=20
  Isynaesthesias = []
  convergences = []
  i = 0
  bw = image_names = [
          'zero.jpg', 'one.jpg', 'two.jpg', 'three.jpg', 'four.jpg',
          'five.jpg', 'six.jpg', 'seven.jpg', 'eight.jpg', 'nine.jpg'
      ]
  for file in os.listdir():
    if file in bw:
      for i in range(3):
        img = cv2.imread(file)
        img = cv2.resize(img, (0,0), fx=0.06, fy=0.06)
        x1 = torch.from_numpy((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()<127).astype('float32'))
        # colour category per pixel 0=original  blue=green
        x2 = torch.from_numpy(np.random.choice([0, 1, 2, 3], size = len(x1)).astype('float32'))
        x2[x1 == 0] = 0

        simulation_emergence_data = torch.stack([x1, x2])
        #syn
        if Izhikevich:
          E_Network = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)

          E_Network.W = weights1
          status, convergence = E_Network.forward(simulation_emergence_data, max_iter = emergence_iterations)
          print('Finalised synaesthetic simulations')
          # non syn
          E_Network_non = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)
          E_Network_non.W = weights1
          E_Network_non.cross_talk = False
          status_n, convergence_n = E_Network_non.forward(simulation_emergence_data, max_iter = emergence_iterations)
          print('Finalised non-synaesthetic simulations')
        else:
          E_Network = GraphemeColourSynaesthesiaNet(np.shape(simulation_emergence_data), M=len(x1)*2)

          E_Network.W = weights1
          status, convergence = E_Network.forward(simulation_emergence_data, max_iter = emergence_iterations)
          print('Finalised synaesthetic simulations')
          # non syn
          E_Network_non = GraphemeColourSynaesthesiaNet(np.shape(simulation_emergence_data), M=len(x1)*2)
          E_Network_non.W = weights1
          E_Network_non.cross_talk = False
          status_n, convergence_n = E_Network_non.forward(simulation_emergence_data, max_iter = emergence_iterations)
          print('Finalised non-synaesthetic simulations')
        # Calculate Synaesthetic Baseline
        Synaesthesia_s = torch.stack([E_Network.s1, E_Network_non.s1])
        Non_Synaesthesia_s = torch.stack([E_Network.s2, E_Network_non.s2])

        Isynaesthesia = torch.mean(abs(Synaesthesia_s -  Non_Synaesthesia_s), 0).mean()  # synaesthesia output current I

        Isynaesthesias.append(Isynaesthesia.detach().numpy())
        print('Synaesthetic Baseline:', Isynaesthesia)
        convergences.append([convergence, convergence_n])
        del E_network, E_network_non, Synaesthesia_s, Non_Synaesthesia_s, convergence, convergence_n
      i+=1
  return Isynaesthesias
def faux_train(Izhikevich=True):
  n = tdistb.Bernoulli(torch.tensor([0.3]))
  x1 = n.sample((9,)).float()[:,0]# modality 2 black or white

  n2 = tdistc.Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
  x2 = n2.sample((9,)).float()# modality 1  range of colours
  x2[x1==0] = 0
  simulation_emergence_data = torch.stack([x1, x2])
  print(np.shape(simulation_emergence_data))
  if Izhikevich:
    E_Network = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)
  else:
    E_Network = GraphemeColourSynaesthesiaNet( np.shape(simulation_emergence_data), M=len(x1)*2)

  #syn
  weights1 = E_Network.W
  emergence_iterations=60
  Isynaesthesias = []
  convergences = []
  
  for i in range(10):
    n = tdistb.Bernoulli(torch.tensor([0.3]))
    x1 = n.sample((9,)).float()[:,0]# modality 2 black or white

    n2 = tdistc.Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    x2 = n2.sample((9,)).float()# modality 1  range of colours
    x2[x1==0] = 0
    simulation_emergence_data = torch.stack([x1, x2]) 
    for j in range(3):
      if j==0:
        print('iter', i)
      #syn
      if Izhikevich:
        E_Network = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)

        E_Network.W = weights1
        status, convergence = E_Network.forward(simulation_emergence_data, max_iter = emergence_iterations)
        print('Finalised synaesthetic simulations')
        # non syn
        E_Network_non = GraphemeColourSynaesthesiaSpikeNet(params, np.shape(simulation_emergence_data), M=len(x1)*2)
        E_Network_non.W = weights1
        E_Network_non.cross_talk = False
        status_n, convergence_n = E_Network_non.forward(simulation_emergence_data, max_iter = emergence_iterations)
        print('Finalised non-synaesthetic simulations')
      else:
        E_Network = GraphemeColourSynaesthesiaNet(np.shape(simulation_emergence_data), M=len(x1)*2)

        E_Network.W = weights1
        status, convergence = E_Network.forward(simulation_emergence_data, max_iter = emergence_iterations)
        print('Finalised synaesthetic simulations')
        # non syn
        E_Network_non = GraphemeColourSynaesthesiaNet(np.shape(simulation_emergence_data), M=len(x1)*2)
        E_Network_non.W = weights1
        E_Network_non.cross_talk = False
        status_n, convergence_n = E_Network_non.forward(simulation_emergence_data, max_iter = emergence_iterations)
        print('Finalised non-synaesthetic simulations')
      # Calculate Synaesthetic Baseline
      s1 = torch.stack([E_Network.s1, E_Network_non.s1])
      s2 = torch.stack([E_Network.s2, E_Network_non.s2])
      # Synaesthesia_v = E_Network.spikes
      # Non_Synaesthesia_v = E_Network_non.spikes
      Isynaesthesia = torch.mean(abs(s1 -  s2), 0).mean()  # synaesthesia output current I

      # Isynaesthesia = torch.mean(Synaesthesia_v -  Non_Synaesthesia_v, 0).mean()  # synaesthesia output current I
      
      Isynaesthesias.append(Isynaesthesia.detach().numpy())
      # print('Synaesthetic Baseline:')
      convergence.append([convergence, convergence_n])
      # del Synaesthesia_s, Non_Synaesthesia_s, convergence, convergence_n
      
  return Isynaesthesias, convergence


Isynaesthesias, convergence= faux_train(False)
torch.save(Isynaesthesias, 'Standard_number_color_Synaesthesia.pt')

print(convergence)

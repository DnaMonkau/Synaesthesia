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
from torchdiffeq import odeint
import cv2
import torch.distributions as tdist
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleSynaesthesiaNet(nn.Module):
  def __init__(self, input_dim, M, tau=1.0, eta=0.1, cross_talk = True):
    super(SimpleSynaesthesiaNet, self).__init__()

    self.modalities = input_dim[0] # Amount of modalities
    self.cross_talk = cross_talk # True or False
    # Neurons
    self.tau =  nn.Parameter(torch.tensor(tau)) # time constant
    self.eta = nn.Parameter(torch.tensor(eta))
    self.input_dim = input_dim
    self.N = input_dim[-1]
    self.M = M

    self.time_step = tau # steps taken in ms
    self.a_1 = [] # dynamics of a_1 in case -
    self.a_2 = [] # dynamics of a_2 in case +
    self.a_2_neg = [] # dynamics of a_2 in case -
    self.critical_eta = None # critical learning rate
    self.variance = None # output variance for modalities

    self.g = nn.Sigmoid()
    # debatable whether M or modalities
    if cross_talk:
      # Generate random angles
      theta = 2 * np.pi * torch.rand(1)

      # Set radius for the ring
      radius = 0.1  # Can be adjusted

      # Convert to cartesian coordinates
      K12 = radius * torch.cos(theta)
      K21 = radius * torch.sin(theta)

      self.K = nn.Parameter(torch.zeros(self.M, self.M) + torch.tensor([K21, K12]), requires_grad=False) # start at 0 fixed point or near 0
      with torch.no_grad():
        self.K.fill_diagonal_(0) # no self connections
    else:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=False) # start at 0 fixed point

    self.W1 = nn.Parameter(torch.randn(int(self.M/2), self.N), requires_grad=True)
    self.W2 = nn.Parameter(torch.randn(int(self.M/2), self.N), requires_grad=True)

    self.s1 = torch.zeros((int(self.M/2)), requires_grad=False) # Membrane potential or output neurons
    self.s2 = torch.zeros((int(self.M/2)), requires_grad=False) # Membrane potential or output neurons
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
    # self.s[0] = self.g(self.W[0] @ x[0] + self.K[0] @ self.s[1])
    # self.s[1] = self.g(self.W[1] @ x[1] + self.K[1] @ self.s[0]) #
    self.s1 = self.g(self.W1 * x[0] + self.K[0][1].clone() * s2)
    self.s2 = self.g(self.W2 * x[1] + self.K[1][0].clone() * s1)
    return

  def feedforward_dynamics(self, x, t, s1, s2, samp_ts):
    '''
    The feedforward dynamics  of the model
    note: original math and paper do not consider modalities
    '''
    s = torch.stack([s1, s2])
    self.s1, self.s2 = odeint(self.dynamics, s, samp_ts[:t+1])[-1]

    return

  def objective_function(self, x, s1, s2):
    '''
    The cost function use for estimating the
    mutual information between the in and output neurons.
    Is used as a loss or cost function

    note: original math and paper do not consider modalities
    does not work with  k = 0
    '''
    K12 = self.K[0][1].clone()
    K21 = self.K[1][0].clone()

    g_p_1 = self.logistic_derivative(self.W1@x[0] + K12 * s2)
    g_p_2 = self.logistic_derivative(self.W2@x[1] + K21 * s1)
    G_1 = torch.diag(g_p_1)
    G_2 = torch.diag(g_p_2)
    if torch.any(self.K != 0):
      phi_1 = torch.linalg.pinv(torch.linalg.pinv(G_1) - K12)
      phi_2 = torch.linalg.pinv(torch.linalg.pinv(G_2) - K21)
      chi_1 = (phi_1 @ self.W1)
      chi_2 = (phi_2 @ self.W2)
    else:
      phi_1 = G_1
      phi_2 = G_2
      chi_1 = phi_1 @ self.W1
      chi_2 = phi_2 @ self.W2

    E1 = - 0.5 * torch.trace(torch.log(chi_1.T @ chi_1))
    E2 = - 0.5 * torch.trace(torch.log(chi_2.T @ chi_2))
    return E1, E2, chi_1, chi_2, G_1, G_2, phi_1, phi_2

  def learning_rule(self, x, chi_1, chi_2, G_1, G_2, phi_1, phi_2):
    ## numerical calculation with autgrad
    G = [G_1, G_2]
    # print(np.shape(torch.linalg.inv(G[0]**(-1) - np.shape(self.W1))
    if torch.any(self.K != 0):
      chi1 = torch.linalg.pinv(torch.linalg.pinv(G[0]) - self.K[1][0].clone()) @ self.W1.clone() # modality one
      chi2 = torch.linalg.pinv(torch.linalg.pinv(G[1])- self.K[0][1].clone()) @ self.W2.clone() # modality two
    else:
      chi1 = G[0] @ self.W1.clone() # modality one
      chi2 = G[1] @ self.W2.clone() # modality two

    E1 = - 0.5 * torch.trace(torch.log(chi_1.T @ chi_1))
    E2 = - 0.5 * torch.trace(torch.log(chi_2.T @ chi_2))
    w_delta1 = - self.eta * torch.autograd.grad(E1, self.W1, retain_graph=True)[0]
    w_delta2 = - self.eta * torch.autograd.grad(E2, self.W2, retain_graph=True)[0]


    with torch.no_grad():
      self.W1 = nn.Parameter(self.W1.clone() + w_delta1)
      self.W2  = nn.Parameter(self.W2.clone() + w_delta2)
    if self.cross_talk:
      # k_delta = - self.eta * torch.autograd.grad(torch.stack([E1, E2]), self.K, retain_graph=True)[0]
      k_delta1  = self.eta * torch.mean(self.s2) * torch.mean(1-2*self.s1)
      k_delta2 = self.eta * torch.mean(self.s1) * torch.mean(1-2*self.s2)
      with torch.no_grad():
        # self.K = nn.Parameter(self.K.clone() + k_delta)
        self.K[0][1] += k_delta1
        self.K[1][0] += k_delta2
        self.K.fill_diagonal_(0)
    return


  def convergence(self, x, s1, s2):
    alpha1 = torch.mean(s1**2)
    alpha2 = torch.mean(s2**2)
    Tr = 4 * alpha1 * alpha2 - alpha1 - alpha2
    Det = - 9/16 +3*alpha1 +3*alpha2 - 4*alpha1**2 - 4 * alpha2**2 - (29/2)*alpha1*alpha2 +18*alpha1*alpha2**2 +18*alpha1**2*alpha2 - 21*alpha1**2*alpha2**2
    self.critical_eta = -(Tr + torch.sqrt(Tr**2 - 4 * Det))/Det
    # gammas = torch.sqrt(1+self.eta*Tr + self.eta**2*Det)
    gamma1 = abs(1+0.5*self.eta*(Tr + torch.sqrt(Tr**2-4*Det)))
    gamma2 = abs(1+0.5*self.eta*(Tr - torch.sqrt(Tr**2-4*Det)))
    # if torch.abs(A_eigenvalues).all() < 1:
    if gamma1 < 1 and gamma2 < 1:
     return True
    # if float(gammas) < 1:
    #   return True
    return False

  def dynamics_a(self):

    a1 = torch.mean(self.s1)
    a2 = torch.mean(self.s2)
    # a1 = a[0]**2 # first differential function for unstable dynamics
    # a2_neg =(36 * a2**4 - 29 * a2**2 + 6 - torch.sqrt(-6 * a2**2 * (2*a2**2 - 1)**3)) / (84 * a2**4 - 72 * a2**2 + 16) # second differential function for unstable dynamics
    # a2 = (36 * a2**4 - 29 * a2**2 + 6 + torch.sqrt(-6 * a2**2 * (2*a2**2 - 1)**3)) / (84 * a2**4 - 72 * a2**2 + 16) # second differential function for unstable dynamics
    # a_1.append(a1)
    # a_2.append(a2)
    # a_2_neg.append(a2_neg)
    alpha1 = a1
    alpha2= a2
    Det = - 9/16 +3*alpha1 +3*alpha2 - 4*alpha1**2 - 4 * alpha2**2 - (29/2)*alpha1*alpha2 +18*alpha1*alpha2**2 +18*alpha1**2*alpha2 - 21*alpha1**2*alpha2**2


    return Det
  def forward(self, x, max_iter=10):
    status = []
    self.timeline = int((max_iter)/self.time_step)
    self.spikes= torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities, self.N ))
    self.losses = torch.tensor(torch.zeros(int((max_iter)/self.time_step),2))
    self.variances = torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities))
    self.critical_etas = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    self.x = x
    orig_ts = torch.linspace(0., int((max_iter)/self.time_step), max_iter)
    samp_ts = orig_ts
    # s1 = self.s1.clone()
    # s2 = self.s2.clone()
    # self.feedforward_dynamics(x, 0, s1, s2, samp_ts)

    for i in tqdm(range(1,int((max_iter/self.time_step)))):
      with torch.no_grad():
        s1 = self.s1.clone()
        s2 = self.s2.clone()
      det = self.dynamics_a()

      if self.convergence(x, s1, s2): # steady-stae
        status.append('Stable')
        self.steady_state(x, s1, s2)
      elif det == 0:
        status.append('Unstable')
      else:
        status.append('Unstable')

      self.feedforward_dynamics(x, i, s1, s2, samp_ts)

      objective_function= self.objective_function(x, s1, s2)
      loss_1, loss_2, chi_1, chi_2, G_1, G_2, phi_1, phi_2 = self.objective_function(x, s1, s2)

      self.learning_rule(x, chi_1, chi_2, G_1, G_2, phi_1, phi_2)


      # if a1.all() == a2.all() or a1.all() == a2_neg.all():
      #   status.append('Unstable')

      # a1, a2, a2_neg = self.dynamics_a(x)
      # if a1.all() == a2.all() or a1.all() == a2_neg.all():
      #   status.append('Unstable')
      with torch.no_grad():
        self.K.fill_diagonal_(0)

      self.losses[i] = torch.stack([loss_1, loss_2])
      loss_1.backward(retain_graph=True)
      loss_2.backward()

      self.optimizer.step()
      self.optimizer.zero_grad()
      self.train()

      with torch.no_grad():
        self.K.fill_diagonal_(0)

      # self.Ks[i] = torch.stack([self.K[0][1], self.K[1][0]])
      self.variances[i,0] = torch.mean(self.s1**2) - torch.mean(self.s1)**2
      self.variances[i,1] = torch.mean(self.s2**2) - torch.mean(self.s2)**2
      self.critical_etas[i] = self.critical_eta
    return status



  #helper functions
  def dynamics(self, t, s):
    # print((self.W1 @ x[0] + self.K[0][1].clone() * s[1]))
    s1, s2 = s
    s1_dt = -s1 + self.g(self.W1 @ self.x[0] + self.K[0][1].clone() * s2)/self.tau
    s2_dt = -s2 + self.g(self.W2 @ self.x[1] + self.K[1][0].clone() * s1)/self.tau
    return torch.stack([s1_dt, s2_dt])
  def logistic_derivative(self,x):
    return self.g(x) * (1 - self.g(x))
  def second_logistic_derivative(self, x):
    return self.logistic_derivative(x) * (1 - 2*self.g(x))
  def third_logistic_derivative(self, x):
    return self.second_logistic_derivative(x)* (1 - 2*self.g(x)) - 2*self.logistic_derivative(x)**2
def run():
  #syn
  syn=[]
  ks = []
  variances = np.linspace(0.01, 0.25, 3)
  # Generate random angles
  theta = 2 * np.pi * torch.rand(1)

  # Set radius for the ring
  radius = 0.1  # Can be adjusted

  # Convert to cartesian coordinates
  K12 = radius * torch.cos(theta)
  K21 = radius * torch.sin(theta)

  plots = []
  plots2 = []
  plots3 = []
  W1 = nn.Parameter(torch.randn(1, 1), requires_grad=True)
  W2 = nn.Parameter(torch.randn(1, 1), requires_grad=True)
  outs =[]
  vs=[]
  for i in range(len(variances)):
    for j in range(len(variances)):

      n = tdist.Normal(0, torch.sqrt(torch.tensor([variances[i]])))
      n2 = tdist.Normal(0, torch.sqrt(torch.tensor([variances[j]])))
      x1 = n.sample((1,)).float()[:,0] # modality 1
      x2 = n2.sample((1,)).float()[:,0] # modality 2 adjust variances for analysis
      x = torch.stack([x1, x2])

      net = SimpleSynaesthesiaNet(np.shape(x), 2)
      # consistent weights
      net.K = nn.Parameter(torch.tensor([[0, K12], [K21,0]]))
      net.W1 = W1
      net.W2 = W2
      # print(np.shape(x), x)
      out = net.forward(x,1000)
      if torch.round(abs(net.K[0][1]), decimals=1) < 1.5 and torch.round(abs(net.K[1][0]), decimals=1) < 1.5:
      # if out[-1] == 'Stable':
        plots.append([variances[i], variances[j], 1]) # no cross-talk present
      # elif out[-1] == "Unstable":
      else:
        if out[-1] == 'Stable' and torch.round(abs(net.K[0][1]) - abs(net.K[1][0]), decimals=1) > 1.5:
          plots3.append([variances[i], variances[j],2])
          print('unsure')
          print(net.K)
        elif out[-1] == 'Stable':
          plots.append([variances[i], variances[j], 1]) # no cross-talk present
          print('unsure stable')
        else:
          plots2.append([variances[i], variances[j], 0])
          print('unstable')
          print(net.K)
      outs.append(out)
      vs.append([variances[i], variances[j]])
      ks.append(net.K)
  # df = pd.DataFrame()
  # pd.DataFrame({ks}).to_csv('EmergentSynaesthesiaks.csv')
  # pd.DataFrame({outs}).to_csv('EmergentSynaesthesiaouts.csv')
  # pd.DataFrame({plots}).to_csv('EmergentSynaesthesiapl.csv')
  # pd.DataFrame({plots2}).to_csv('EmergentSynaesthesiapl2.csv')
  # pd.DataFrame({vs}).to_csv('EmergentSynaesthesiavs.csv')

  return plots, plots2, plots3
def make_fig(plots, plots2, plots3):
  for i in range(len(plots)):
    plt.scatter(plots[i][0], plots[i][1], color ='green')
  for i in range(len(plots2)):
    plt.scatter(plots2[i][0], plots2[i][1], color = 'red')
  for i in range(len(plots3)):
    plt.scatter(plots3[i][0], plots3[i][1], color = 'orange')    
  plt.savefig('../Synaesthesia/out.jpeg')
  # plt.xlim([0, 0.25])
  # plt.ylim([0, 0.25])
  # plt.title()
  plt.xlabel('Varianance unit #1')
  plt.ylabel('Varianance unit #2')
plots, plots2, plots3 = run()
make_fig(plots, plots2, plots3)

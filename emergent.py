
from itertools import product
import numpy as np
from colorsys import hsv_to_rgb
import scipy.stats
import matplotlib.pyplot as plt
# from torch.distributions import normal as tdist
import torch
import torch.nn as nn
from tqdm import tqdm
from random import random
import scipy
import os
import imageio
import seaborn as sns
import scipy
# !pip install torchdiffeq
# from torchdiffeq import odeint
#pip install opencv-python
import cv2
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = '../Synaesthesia/WorkingMemory/images/original'

def numpy2hsv(array):
  rgbimg = np.zeros((len(array), 3))
  for i in range(len(array)):
    rgbimg[i] = hsv_to_rgb(array[i],1,1)
  return rgbimg*255
def create_rgb_to_bin(N, colour_set=(np.array([0, 0.33, 0.66, 0.99])*255).astype(int)):
  chunk = N //3
  colours = []
  rgb_bin = []
  for r in colour_set:
    for g in colour_set:
      for b in colour_set:
        if [r,g,b] not in colours:
          colours.append([r,g,b])
          r_chunk = list(str(bin(r)[2:]).zfill(chunk))
          g_chunk = list(str(bin(g)[2:]).zfill(N-chunk*2))
          b_chunk = list(str(bin(b)[2:]).zfill(chunk))
          rgb_bin.append(np.array(r_chunk+  g_chunk+ b_chunk).astype(int))

  return rgb_bin, colours

#############################################################################################################################
class GraphemeColourSynaesthesiaNet(nn.Module):
  def __init__(self, input_dim, M, max_iter  = 10, tau=1.0,tolfun=4e-005, eta=0.01, eta_w =0.01, cross_talk = True, modalities = 2, FF = True):
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
    if cross_talk:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=True) # start at 0 fixed point or near 0

      # # Generate random angles
      # theta = 2 * np.pi * torch.rand(self.M**2)

      # # Set radius for the ring
      # radius = 0.1  # Can be adjusted

      # # Convert to cartesian coordinates
      # K12 = radius * torch.cos(theta)
      # with torch.no_grad():
      #   self.K = nn.Parameter(torch.reshape(K12, (self.M, self.M)))
            # self.K[j][i]= K21
      with torch.no_grad():
        self.K.fill_diagonal_(0) # no self connections
    else:
      self.K = nn.Parameter(torch.zeros(self.M, self.M), requires_grad=False) # start at 0 fixed point
    # Initialise feedforward weights
    if self.FF:
      self.W = nn.Parameter(torch.empty(self.M, self.N).uniform_(0,0.6), requires_grad=True)
    else:
      self.W = nn.Parameter(torch.randn(self.M, self.N), requires_grad=False)

    self.s1 = torch.zeros(self.M//2, requires_grad=False) # Membrane potential or output neurons
    self.s2 = torch.zeros(self.M//2, requires_grad=False) # Membrane potential or output neurons
    self.optimizer = torch.optim.Adam(self.parameters(), lr=eta)
    self.timeline = int((max_iter)/self.time_step)
    self.spikes= torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities, self.N ))
    self.losses = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    self.variances = torch.tensor(torch.zeros(int((max_iter)/self.time_step), self.modalities))
    self.critical_etas = torch.tensor(torch.zeros(int((max_iter)/self.time_step)))
    assert self.N <= self.M, 'This is an overcomplete network use more output than input neurons'
  def steady_state(self, x, s1, s2):
    '''
    The steady state interactions
    '''
    # Apply first order activities to the output perception s
    s = torch.stack([self.s2, self.s1]).flatten()
    s = self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s))
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    return self.s1, self.s2

  def feedforward_dynamics(self, x, t, s1, s2, samp_ts):
    '''
    The feedforward dynamics  of the model
    note: original math and paper do not consider modalities
    '''
    s = torch.stack([s2, s1]).flatten()
    # try:
    # s = odeint(self.dynamics, s, samp_ts[:t+1])[-1]
    s = self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s))
    self.s1, self.s2 = s[:self.M//2], s[self.M//2:]
    # except:
      # self.s1, self.s2 = s1, s2
    return

  def objective_function(self, x, E,  s1, s2):
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
      chi_mult = torch.log(chi.T@chi)
      # if torch.isnan(chi_mult).any():
      #   E = E + 1e2
      # else:
      chi_mult = torch.where(torch.isnan(chi_mult),1.0, chi_mult)
      # chi_mult = torch.where(chi_mult<1e7,torch.exp(chi_mult), chi_mult)
      E = - 0.5 * torch.trace(torch.log(torch.exp(chi_mult)))

      # E2 = - 0.5 * torch.trace(torch.log(chi_2.T @ chi_2))
      return E, chi, G, phi

  def learning_rule(self, x,chi, G, phi, E):
    if self.FF:
      w_delta = - self.eta_w * torch.autograd.grad(E, self.W, retain_graph=True)[0]
      with torch.no_grad():
        self.W = nn.Parameter(self.W.clone() + w_delta)

    if self.cross_talk:
      # k_delta = self.eta_k *torch.mean((chi@G).T + phi.T)
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

  def forward(self, xs, max_iter=10, d=True):
    status = []
    loss = torch.tensor([0.], requires_grad=True)
    iterations = max_iter // self.time_step
    old_K = self.K.clone()
    self.Ks = []

    # In case the data is concatenated split iterations
    if d==True:

      index = int(iterations // len(xs) )+1

    samp_ts = torch.linspace(0., int((max_iter)/self.time_step), int(max_iter/self.time_step))
    converged = []
    s1_prev, s2_prev = -np.inf, -np.inf

    for i in tqdm(range(1,int(iterations))):
      # use the proper data based on the split
      old_loss = loss.clone()
      x = xs[int(i//index)-1].flatten()
      self.x = x
      with torch.no_grad():
        s1 = self.s1.clone()
        s2 = self.s2.clone()
        self.Ks.append(self.K.clone())

        if i == iterations//10:
          self.FF = False
        #   plt.imshow(np.reshape(self.x[:64].detach().numpy(),(8,8)))
        #   plt.show()
        if (abs(self.K) == 0).all():
          status.append('Stable')
          s1, s2 = self.steady_state(x, s1, s2)
        else:
          status.append('Unstable')
          self.feedforward_dynamics(x, i, s1, s2, samp_ts)
          self.FF = False

      # self.s2 = self.colour_cat(self.s2)
      # self.s2 = self.colour_bin(self.s2)
      loss, chi, G, phi = self.objective_function(x, loss, s1, s2)
      if self.convergence( s1_prev, s2_prev, s1, s2, loss, i): # steady-stae
        converged.append(i)
        print('converged')
        break
      self.learning_rule(x,chi, G, phi, loss)
      self.losses[i] = loss
      loss.backward()

      self.optimizer.step()
      self.optimizer.zero_grad()
      if torch.isnan(self.K).any():
        self.K = nn.Parameter(old_K.clone())
      self.stable(x, s1, s2)

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
    self.s2 = self.rgb_bin_closeness(len(self.s2), self.s2)[0]
    return status, converged
  def train(self, x, max_iter):
    status, converged = self.forward(x, max_iter,  d=True)
    return status, converged
  def predict(self, x, s):
    s1, s2 = torch.reshape(s,  (2,self.M//2))
    s = torch.stack([s2, s1]).flatten()
    s = self.g(torch.matmul(self.W, x) + torch.matmul(self.K, s.T))
    s[self.M//2:] = s[self.M//2:]
    return s

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
  def colour_cat(self, tensor, colours = torch.tensor([0, 0.33, 0.66, 0.99])):
    cat_array =[]
    for i in range(len(tensor)):
      index = torch.argmin(abs(tensor[i] - colours))
      cat_array.append(colours[index])
    return torch.tensor(cat_array)

  def rgb_bin_closeness(self, N, array, colour_set=torch.tensor([0, 0.33, 0.66, 0.99])*255, dtype =int):
    array = array.detach().numpy()
    chunk = N //3
    r = array[:N//3].astype(int)
    g = array[N//3:-N//3].astype(int)
    b = array[-N//3:].astype(int)
    r_dec = int(''.join(map(str,(r[-8:]))),2)
    g_dec = int(''.join(map(str,(g[-8:]))),2)
    b_dec = int(''.join(map(str,(b[-8:]))),2)
    r  = int(colour_set[torch.argmin(abs(colour_set - r_dec))].item())
    g  = int(colour_set[torch.argmin(abs(colour_set - g_dec))].item())
    b  = int(colour_set[torch.argmin(abs(colour_set - b_dec))].item())
    r_chunk = list(str(bin(r)[2:]).zfill(chunk))
    g_chunk = list(str(bin(g)[2:]).zfill(N-chunk*2))
    b_chunk = list(str(bin(b)[2:]).zfill(chunk))
    arr = np.array([r_chunk+g_chunk+b_chunk]).astype(int)
    array = torch.tensor(arr)

    return array
#########################################################################################################
def train(emergence_iterations=50):
  img = cv2.imread(os.path.join(PATH, 'zero.jpg'))
  img = cv2.resize(img, (0,0), fx=0.10, fy=0.10)
  # flat gray scale number array
  x1 = torch.from_numpy((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()).astype('float32'))
  # normalise
  x1 = x1/255
  # colour category per pixel 0=original  blue=green
  # x2 = torch.from_numpy(np.random.choice([0, 0.33, 0.66, 0.99], size = len(x1)).astype('float32'))
  rgb_bin, _ = create_rgb_to_bin(len(x1))
  x2 = torch.from_numpy(rgb_bin[0])

  x2 = torch.from_numpy(rgb_bin[0])

  simulation_emergence_data = torch.stack([x1, x2])
  E_Network = GraphemeColourSynaesthesiaNet(np.shape(simulation_emergence_data), M=len(x1)*2, max_iter=emergence_iterations)
  #syn
  weights1 = E_Network.W
  Isynaesthesias = []
  convergences = []
  # bw = image_names = [
  #         'zero.jpg', 'one.jpg', 'two.jpg', 'three.jpg', 'four.jpg',
  #         'five.jpg', 'six.jpg', 'seven.jpg', 'eight.jpg', 'nine.jpg'
  #     ]
  x = []
  for i in range(3): # train on all colour grapheme combinations
    j = 0
    random.shuffle(rgb_bin)

    for file in os.listdir(PATH):
        if 'jpg'  in file:
            img = cv2.imread(os.path.join(PATH, file))
            img = cv2.resize(img, (0,0), fx=0.10, fy=0.10)
            x1 = torch.from_numpy((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()).astype('float32'))
            # normalise
            x1 = x1<127
            # colour category per pixel 0=original  blue=green
            # x2 = torch.from_numpy(np.random.choice(range(0,360), size = len(x1)).astype('float32'))
            x2 = torch.from_numpy(rgb_bin[j].astype(np.float32))
            # normalise

            simulation_emergence_data = torch.stack([x1, x2])
            x.append(simulation_emergence_data)
            j+=1
  x = torch.stack(x)
  shuffled_indices = torch.randperm(x.shape[0])

  x = x[shuffled_indices]

  x = np.array(x)
  x = torch.from_numpy(x)
  status, converged = E_Network.train(x, emergence_iterations)
  return E_Network, status, converged
def run():
  x = []
  col= []
  col_rand = []
  # bw = image_names = [
  #         'zero.jpg', 'one.jpg', 'two.jpg', 'three.jpg', 'four.jpg',
  #         'five.jpg', 'six.jpg', 'seven.jpg', 'eight.jpg', 'nine.jpg'
  #     ]
  for file in os.listdir(PATH):
      if 'jpg' in file:
        img = cv2.imread(os.path.join(PATH, file))
        img = cv2.resize(img, (0,0), fx=0.10, fy=0.10)
        x1 = torch.from_numpy((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()).astype('float32'))
        # normalise
        x1 = x1/255
        # colour category per pixel 0=original  blue=green
        x2 = torch.zeros(len(x1))
        simulation_emergence_data = torch.stack([x1, x2])
        x.append(simulation_emergence_data)
  chunk = int(len(x2)/3) # a chunk for every channel, rgb = 3
  S = 3000 # sessions
  for i in range(S):
    colour_random_mean = []
    colour_predictionhsv_median = []
    colour_predictionhsv_mean = []
    colours = []

    # train
    Net, status, converged = train(10)
    # Run predictions per grapheme
    for j in range(len(x)):
      xs = torch.from_numpy(np.array(x[j])).flatten()
      xs[64:] = torch.from_numpy(np.zeros(len(x1)).astype('float32')) # set colour to 0
      s = torch.zeros(np.shape(xs))
      s1, s2= torch.reshape(Net.predict(xs, s), (2,64))
      # Assign categorical colors 1 out of 81
      s2 = Net.rgb_bin_closeness(64,s2)[0]
      s1 = s1.detach().numpy()
      s2 = s2.detach().numpy()
      # Calculate decimal value per binary code 
      r = (s2[:chunk]).astype(int)
      g =(s2[chunk: -chunk]).astype(int)
      b =(s2[-chunk:]).astype(int)
      re = int(''.join(map(str,(r[-8:]))),2)
      ge = int(''.join(map(str,(g[-8:]))),2)
      be = int(''.join(map(str,(b[-8:]))),2)
      # Add rgb values
      rgb_space = np.array([re, ge, be])
      # Create random vector
      random_vector = torch.tensor(np.random.choice([0, 1], size=len(x1)))
      # Calculate decimal value per binary code 
      random_vector = (Net.rgb_bin_closeness(64,random_vector)[0]).detach().numpy()
      random_r = (random_vector[:chunk]).astype(int)
      random_g =(random_vector[chunk: -chunk]).astype(int) 
      random_b =(random_vector[-chunk:]).astype(int)
      rr = int(''.join(map(str,(random_r[-8:]))),2)
      gr = int(''.join(map(str,(random_g[-8:]))),2)
      br = int(''.join(map(str,(random_b[-8:]))),2)
      # Add rgb values
      rand_rgb_space = np.array([rr, gr, br])

      colour_predictionhsv_mean.append(rgb_space)
      colour_random_mean.append(rand_rgb_space)

    col.append(colour_predictionhsv_mean)
    col_rand.append(colour_random_mean)
  return col, col_rand
def colour_in_grapheme(col, col_rand):
  i=0
  HSV_rgb2 = []
  HSV_random = []
  rgb = []
  randoms = []
  for file in os.listdir(PATH):
    if 'jpg'  in file:
      # Get emergent colours per grapheme over all sessions
      column = np.array(col)[:,i]
      
      # Get most common colour excluding black
      # column = [c for c in np.array(col)[:,i] if not np.array_equal(c, np.array([0,0,0]))]
      # Get most common colour excluding black
      # if column: # Check if column is not empty after removing [0,0,0]
        # rgb_mode = scipy.stats.mode(column)[0]
      # else:
        # rgb_mode = scipy.stats.mode(np.array(col)[:,i])[0]
      
      rgb_mode = scipy.stats.mode(column)[0]
      # Get random colours per grapheme over all sessions
      column = np.array(col_rand)[:,i]

      # Get most common colour excluding black
      # column = [c for c in np.array(col_rand)[:,i] if not np.array_equal(c, np.array([0,0,0]))]
      # if column: # Check if column is not empty after removing [0,0,0]
        # random_mode = scipy.stats.mode(column)[0]
      # else:
        # random_mode = scipy.stats.mode(np.array(col)[:,i])[0]
      # Get most common colour
      random_mode = scipy.stats.mode(column)[0]
      
      ## HSV transformation for hue histogram
      # hsv_rgb = cv2.cvtColor(np.array(rgb_mean* 255).astype(np.uint8).reshape(1, 1, 3),cv2.COLOR_RGB2HSV)[0,0]
      # hsv_random = cv2.cvtColor(np.array(random_mean* 255).astype(np.uint8).reshape(1, 1, 3),cv2.COLOR_RGB2HSV)[0,0]
      # HSV_rgb2.append(hsv_rgb)
      # HSV_random.append(hsv_random)

      img = cv2.imread(os.path.join(PATH, file))
      # Add colour to black space in grapheme, a.k.a pattern
      img_c = np.where(img< 127, rgb_mode, 255)
      img_r = np.where(img< 127, random_mode, 255)

      # cv2_imshow(img_c)
      # plt.show()
      # cv2_imshow(img_r)
      # plt.show()
      rgb.append(rgb_mode)
      randoms.append(random_mode)
      cv2.imwrite('../Synaesthesia/new_res/'+'emergent_colour_'+file, img_c)
      cv2.imwrite('../Synaesthesia/new_res/'+'trivial_colour_'+file, img_r)
      i+=1
  # zip -r res.zip res/
  return rgb, randoms
def emergent_stats(rgb, randoms):
  channels = ['red', 'green', 'blue']
  sns.set_palette(sns.diverging_palette(145, 300, s=60, n=3))
  for i, colour in enumerate(channels):
    sns.histplot(np.array(rgb)[:,i] , bins=50, kde=True, label = '(emergent) '+colour)
  plt.legend()
  plt.title('Emergent colour per grapheme in RGB space histogram')
  plt.savefig('rgb_hist_col_graph.jpg')
  plt.show()

  for i, colour in enumerate(channels):
    sns.histplot(np.array(randoms)[:,i] , bins=50, kde=True,  label = '(random) '+colour)
  plt.legend()
  plt.title('Randomly generated colour per grapheme in RGB space histrogram') 
  plt.savefig('random_hist_col_graph.jpg')
  plt.show()
  with open("stats_emer.txt", "a") as f:
    f.write("Mann-Whitney {mann:}\nT-test{ttest} \nAnova {anova}".format(mann=scipy.stats.mannwhitneyu(rgb, randoms), 
                                         ttest = scipy.stats.ttest_ind(rgb, randoms), 
                                         anova = scipy.stats.f_oneway(rgb, randoms)))

###################################################################################################
col, col_rand = run()
rgb, randoms = colour_in_grapheme(col, col_rand)
emergent_stats(rgb, randoms)

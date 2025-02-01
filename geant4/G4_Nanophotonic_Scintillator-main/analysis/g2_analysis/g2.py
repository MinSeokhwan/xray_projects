import numpy as np
import sys
sys.path.insert(0, '../../')
import analysis.read_simulation as rs
import random
import math
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def simulate_electron_emission(mean_rate, total_time, loss_probability_electron=0):
    """
    Thermal gun mode
    """
    time = 0
    emission_times = []
    while time < total_time:
        interarrival_time = random.expovariate(mean_rate)
        time += interarrival_time
        if time < total_time and random.random() >= loss_probability_electron:
            emission_times.append(time)

    return emission_times


def simulate_photons_emission_when_electron_hits(electron_times, photon_emission_rate, time_variation_photons_rate=0.9, loss_probability=0):
    photon_times = np.array([])
    photons_times_after_lossy_medium = []
    for electron_time in electron_times:
        num_photons = np.random.poisson(1)

        cur_photon_times = np.random.exponential(time_variation_photons_rate, size=num_photons) + electron_time
        # cur_photon_times = np.random.exponential(time_variation_photons_rate, size=1) + electron_time
        # adding loss mask
        cur_photon_times = [photon_time for photon_time in cur_photon_times if random.random() >= loss_probability]

        photon_times = np.concatenate((photon_times, cur_photon_times))

    return photon_times

## gaussian experiment
def simulate_gaussian_photons_emission_when_electron_hits(electron_times, photon_emission_rate, time_variation_photons_rate=0.9, loss_probability=0):
    photon_times = np.array([])
    photons_times_after_lossy_medium = []
    for electron_time in electron_times:
        num_photons = np.random.poisson(photon_emission_rate)

        cur_photon_times = np.random.normal(electron_time, time_variation_photons_rate, size=num_photons)
        # adding loss mask
        cur_photon_times = [photon_time for photon_time in cur_photon_times if random.random() >= loss_probability]

        photon_times = np.concatenate((photon_times, cur_photon_times))

    return photon_times



def time_binary_search(times, current_looked_idx, cutoff_time=25):
  """
  Purpose: find window in log(n)
  :returns tuple: (s_idx, e_idx) two indices, one is the starting point and the second is the ending
  """
  # print(times)
  s_idx = 0
  e_idx = len(times) - 1
  previous_s_idx = 0
  examined_time = times[current_looked_idx]
  s_half = math.ceil(current_looked_idx/2)
  e_half = math.ceil((e_idx-current_looked_idx)/2)

  # s_idx
  while True:
    try:
      tau_from_sidx = examined_time - times[s_idx]
    except Exception as e:
      print("###### s_idx: ", s_idx)
    if s_idx:
      tau_from_sidx_minus_one = examined_time - times[s_idx-1]
      if tau_from_sidx <= cutoff_time and tau_from_sidx_minus_one > cutoff_time:
        break
      elif tau_from_sidx > cutoff_time:
        s_idx = s_idx + s_half
      elif tau_from_sidx_minus_one <= cutoff_time:
        s_idx = s_idx - s_half
      # check edge cases
      if s_idx < 0:
        s_idx = 0
      elif s_idx > len(times) - 1:
        s_idx = len(times) - 1
      s_half = math.ceil(s_half/2)
    else:
      if tau_from_sidx <= cutoff_time:
        break
      s_idx =+ s_half
      s_half = math.ceil(s_half/2)

  flag = 0

  # e_idx
  while True:
    try:
      tau_from_eidx = times[e_idx] - examined_time
    except Exception as e:
      print(e, "e_idx:", e_idx, ", times:", len(times))
      print(times)
    if e_idx < len(times) - 1:
      tau_from_eidx_plus_one =  times[e_idx+1] - examined_time
      if tau_from_eidx <= cutoff_time and tau_from_eidx_plus_one > cutoff_time:
        break
      elif tau_from_eidx > cutoff_time:
        e_idx = e_idx - e_half
      elif tau_from_eidx_plus_one <= cutoff_time:
       e_idx = e_idx + e_half
       if e_idx > len(times) - 1:
        e_idx = len(times) - 1
        # flag = 1
      e_half = math.ceil(e_half/2)
    else:
      if tau_from_eidx <= cutoff_time:
        break
      e_idx = e_idx - e_half
      e_half = math.ceil(e_half/2)
  if flag:
    print("times[current_looked_idx] : ", times[current_looked_idx], "; times[e_idx]" , times[e_idx])
  return s_idx, e_idx

def one_process(mean_electron_rate=0.4, total_time=10000, photon_emission_rate=10, time_variation_photons_rate=0.9, probability_photon=0, loss_probability_electron=0, gaussian=0):
  total_time = total_time  # Total simulation time
  photon_emission_rate = photon_emission_rate

  emission_times = simulate_electron_emission(mean_electron_rate, total_time, loss_probability_electron=loss_probability_electron)
  if gaussian:
    photon_times = simulate_gaussian_photons_emission_when_electron_hits(emission_times, photon_emission_rate, time_variation_photons_rate, loss_probability=probability_photon)
  else:
    photon_times = simulate_photons_emission_when_electron_hits(emission_times, photon_emission_rate, time_variation_photons_rate, loss_probability=probability_photon)

  print("Emission times:")
  print("Amount of electrons: ",len(emission_times))
  print("Amount of photons: ", len(photon_times))

  photon_times_list = photon_times.tolist()
  photon_times_list.sort()
  return photon_times_list

def two_processes():
  first = one_process(mean_electron_rate=0.3, total_time=10000, photon_emission_rate=10, time_variation_photons_rate=0.5)
  second = one_process(mean_electron_rate=0.2, total_time=10000, photon_emission_rate=10, time_variation_photons_rate=3)
  together = first + second
  together.sort()
  return together


def create_g2_function(photon_detector_events):
  taus = []
  CUTOFF = 100 # = 10 / xray_rate

  # g(2)
  for i in range(len(photon_detector_events)):
    s_idx, e_idx = time_binary_search(photon_detector_events, current_looked_idx=i, cutoff_time=CUTOFF)
    for j in range(s_idx, e_idx+1):
      if i != j:
        tau = photon_detector_events[i] - photon_detector_events[j]
        taus.append(tau)
  return taus

def get_g_2_0(taus):
  N_bins = 500
  counts, bin_edges = np.histogram(taus, bins=N_bins)
  x = (bin_edges[:-1] + bin_edges[1:]) / 2

  # Define the range of the tail (last 2 seconds)
  tail_range = 1

  # Determine which bins are in the last 2 seconds at each tail
  tail_bins_left = x < x.min() + tail_range
  tail_bins_right = x > (x.max() - tail_range)

  # Calculate the mean count in these tail bins
  mean_tail_count = np.mean(np.concatenate([counts[tail_bins_left], counts[tail_bins_right]]))

  # Normalize by this mean tail count
  counts_normalized = counts / mean_tail_count
  return counts_normalized[int(N_bins/2)]


def compute_g2(photon_detector_events, draw=False):
    photon_detector_events.sort()
    taus = create_g2_function(photon_detector_events)
    if draw:
      draw_g_2_function(taus)
    return get_g_2_0(taus)


def draw_g_2_function(taus, n_bins = 500):
  counts, bin_edges = np.histogram(taus, bins=n_bins)
  x = (bin_edges[:-1] + bin_edges[1:]) / 2

  # Define the range of the tail (last 2 seconds)
  tail_range = 1

  # Determine which bins are in the last 2 seconds at each tail
  tail_bins_left = x < x.min() + tail_range
  tail_bins_right = x > (x.max() - tail_range)

  # Calculate the mean count in these tail bins
  mean_tail_count = np.mean(np.concatenate([counts[tail_bins_left], counts[tail_bins_right]]))

  # Normalize by this mean tail count
  counts_normalized = counts / mean_tail_count

  plt.plot(x, counts_normalized)
  plt.grid(True)
  plt.show()
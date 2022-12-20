import matplotlib.pyplot as plt
import numpy as np
import argparse

# Create an argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=4, help="Number of beams to track.")
parser.add_argument("--version", type=int, default=0, help="File version.")
args = parser.parse_args()

ref = np.load(f"../data/loss_reference_k{args.k}_v{args.version}.npz")["loss"]
haware = np.load(f"../data/loss_InferenceRPUConfig_k{args.k}_v{args.version}.npz")["loss"]
analog = np.load(f"../data/loss_GokmenVlasovPreset_k{args.k}_v{args.version}.npz")["loss"]
plt.plot(np.sort(ref), np.cumsum(ref) / np.sum(ref), linewidth=2)
plt.plot(np.sort(haware), np.cumsum(haware) / np.sum(haware), linewidth=2)
plt.plot(np.sort(analog), np.cumsum(analog) / np.sum(analog), linewidth=2)
plt.grid(alpha=0.4)
plt.xlabel('Misalignment loss (in dB)')
plt.ylabel('CDF')
plt.xlim((-0.1, 5))
plt.legend(['PyTorch', 'Hardware Aware', 'Analog'])
plt.savefig('results.png', dpi=200, bbox_inches='tight')
plt.show()
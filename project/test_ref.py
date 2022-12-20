"""
@organization: New York University
@author: Panagiotis Skrimponis

@copyright: 2022
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Create an argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--clip_snr", type=bool, default=False, help="Clip maximum SNR.")
parser.add_argument("--snr_max", type=float, default=35, help="Maximum value of SNR.")
parser.add_argument("--k", type=int, default=4, help="Number of beams to track.")
parser.add_argument("--version", type=int, default=0, help="Number of beams to track.")
args = parser.parse_args()


def create_sequence(data, num_seq):
    inputs, targets = [], []
    for traj in range(num_trajs):
        data_j = data[traj]
        input_j, target_j = [], []
        for i in range(len(data_j) - num_seq - 1):
            input_j += [data_j[i:i + num_seq]]
            target_j += [data_j[i + num_seq]]
        inputs += input_j
        targets += target_j
    return np.array(inputs), np.array(targets)


class BeamTrackingDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.from_numpy(self.target[idx]).float()


class LSTMModel(nn.Module):
    """

    """

    def __init__(self, device='cpu', batch_size=100, num_beams=17, k=4):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTMCell(num_beams, 128)
        self.lstm2 = nn.LSTMCell(128, 256)
        self.lstm3 = nn.LSTMCell(256, 128)
        self.fc = nn.Linear(128, num_beams)
        self.activation = nn.ReLU()
        self.device = device
        self.k = k
        self.mask = torch.ones(batch_size, num_beams, dtype=torch.float).to(self.device)

    def forward(self, x):
        self.mask = torch.zeros(x.size(0), num_beams, dtype=torch.float).to(self.device)
        self.mask[:, torch.randperm(num_beams)[:self.k]] = 1
        h_t1 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        c_t1 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        h_t2 = torch.zeros(x.size(0), 256, dtype=torch.float).to(self.device)
        c_t2 = torch.zeros(x.size(0), 256, dtype=torch.float).to(self.device)
        h_t3 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        c_t3 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)

        x = torch.swapaxes(x, 0, 1)

        for i in range(x.size()[0]):
            h_t1, c_t1 = self.lstm1(x[i] * self.mask, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.fc(h_t3)
            output = self.activation(output)
            self.mask = (torch.argsort(output, axis=1) < self.k).float()
        return output


data = np.load("../data/beamtracking_dataset_v0.npz")["data"]
num_seq = 5
num_epochs = 5
batch_size = 100
num_trajs = data.shape[0]
num_beams = data.shape[2]
num_samp = data.shape[1] - num_seq - 1

inputs, targets = create_sequence(data, num_seq)

if args.clip_snr:
    inputs[inputs > args.snr_max] = args.snr_max
    targets[targets > args.snr_max] = args.snr_max

max_val = np.max(inputs)
min_val = np.min(inputs)

inputs = (inputs - min_val) / (max_val - min_val)
targets = (targets - min_val) / (max_val - min_val)

num_train_samp = int(0.8 * inputs.shape[0])
train_inputs = inputs[:num_train_samp]
test_inputs = inputs[num_train_samp:]

train_targets = targets[:num_train_samp]
test_targets = targets[num_train_samp:]

train_dataset = BeamTrackingDataset(train_inputs, train_targets)
test_dataset = BeamTrackingDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LSTMModel(device=device, batch_size=batch_size, num_beams=num_beams, k=args.k)
model = model.to(device)

if device == "cuda:0":
    model = nn.DataParallel(model)
    cudnn.benchmark = True

model.load_state_dict(torch.load(f'../data/trained_models/reference_v{args.version}.ckpt'))

# Train
pred_snr = []
best_snr = []
model.eval()
t0 = datetime.now()
with torch.no_grad():
    for batch_index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        pred = torch.argmax(outputs, axis=1)
        best = torch.argmax(targets, axis=1)

        pred = pred.to('cpu').numpy()
        best = best.to('cpu').numpy()
        for batch_index_0, beam_index in enumerate(pred):
            pred_snr.append(targets[batch_index_0, beam_index].to('cpu').numpy())

        for batch_index_1, beam_index in enumerate(best):
            best_snr.append(targets[batch_index_1, beam_index].to('cpu').numpy())
        print(f"\rElapsed Time: {datetime.now() - t0}, Progress: {(batch_index + 1) * 100 / len(test_loader):.2f} %",
              end="")
print("")
best_snr = np.array(best_snr) * (max_val - min_val) + min_val
pred_snr = np.array(pred_snr) * (max_val - min_val) + min_val

misalign_loss = best_snr - pred_snr

np.savez_compressed("../data/misalign_loss_ref.npz", loss=misalign_loss)
plt.plot(np.sort(misalign_loss), np.cumsum(misalign_loss) / np.sum(misalign_loss), linewidth=2)
plt.grid(alpha=0.4)
plt.xlabel('Misalignment loss (in dB)')
plt.ylabel('CDF')
plt.xlim((-0.1, 5))
plt.show()
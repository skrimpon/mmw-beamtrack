import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# Create an argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--clip_snr", type=bool, default=False, help="Clip maximum SNR.")
parser.add_argument("--snr_max", type=float, default=35, help="Maximum value of SNR.")
parser.add_argument("--k", type=int, default=4, help="Number of beams to track.")
parser.add_argument("--version", type=int, default=0, help="File version.")
args = parser.parse_args()

def create_sequence(data, num_seq):
    inputs, targets = [], []
    for traj in range(num_trajs):
        data_j = data[traj]
        input_j, target_j = [], []
        for i in range(len(data_j) - num_seq - 1):
            input_j += [data_j[i:i+num_seq]]
            target_j += [data_j[i+ num_seq]]
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
    def __init__(self, device, batch_size=100, num_beams=17, k=4):
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
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    train_loss = []
    val_loss = []

    # Train
    num_batch = len(train_loader)
    model.train()
    t0 = datetime.now()
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f'\r\tTrain Time: {datetime.now() - t0}, Progress: {(batch_index + 1) / num_batch * 100:.2f} %', end='')
    print("")

    # Validation
    num_batch = len(test_loader)
    t1 = datetime.now()
    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
            print(f'\r\tVal Time: {datetime.now() - t1}, Progress: {(batch_index + 1) / num_batch * 100:.2f} %', end='')
    scheduler.step()
    print(
        f'\n\tEpoch Time: {datetime.now() - t0}, Train Loss: {np.mean(train_loss):.3f}, Val Loss: {np.mean(val_loss):.3f}')

torch.save(model.state_dict(), f'../data/trained_models/reference_k{args.k}_v{args.version}.ckpt')
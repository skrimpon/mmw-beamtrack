import torch
import argparse
import numpy as np
import torch.nn as nn
from os.path import exists
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from aihwkit.nn import AnalogLinear
from aihwkit.nn import AnalogLSTMCell
from aihwkit.nn import AnalogSequential

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.simulator.configs import InferenceRPUConfig

from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.inference import GlobalDriftCompensation
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.simulator.configs.utils import WeightClipType
from aihwkit.simulator.configs.utils import WeightModifierType

# Create an argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--rpu', type=int, default=0,
                    help='Select between analog or hardware-aware training (i.e., 0 or 1)')
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


class AnalogLSTMModel(AnalogSequential):
    def __init__(self, rpu_config, device, k=2):
        super(AnalogLSTMModel, self).__init__()
        self.lstm1 = AnalogLSTMCell(17, 128, rpu_config=rpu_config, bias=True)
        self.lstm2 = AnalogLSTMCell(128, 256, rpu_config=rpu_config, bias=True)
        self.lstm3 = AnalogLSTMCell(256, 128, rpu_config=rpu_config, bias=True)
        self.fc = AnalogLinear(128, 17, rpu_config=rpu_config)
        self.activation = nn.ReLU()
        self.device = device
        self.k = k

    def forward(self, x):
        # Initialize mask
        self.mask = torch.zeros(x.size(0), num_beams, dtype=torch.float).to(self.device)
        self.mask[:, torch.randperm(num_beams)[:self.k]] = 1

        # Initialize hidden states of LSTMs
        h_t1 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        c_t1 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        h_t2 = torch.zeros(x.size(0), 256, dtype=torch.float).to(self.device)
        c_t2 = torch.zeros(x.size(0), 256, dtype=torch.float).to(self.device)
        h_t3 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)
        c_t3 = torch.zeros(x.size(0), 128, dtype=torch.float).to(self.device)

        x = torch.swapaxes(x, 0, 1)

        for i in range(x.size()[0]):
            _, (h_t1, c_t1) = self.lstm1(x[i] * self.mask, (h_t1, c_t1))
            _, (h_t2, c_t2) = self.lstm2(h_t1, (h_t2, c_t2))
            _, (h_t3, c_t3) = self.lstm3(h_t2, (h_t3, c_t3))
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

if args.rpu == 0:
    output_file = 'GokmenVlasovPreset'
    rpu_config = GokmenVlasovPreset()
elif args.rpu == 1:
    output_file = 'InferenceRPUConfig'
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.out_res = -1.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.pdrop = 0.03
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    rpu_config.modifier.std_dev = 0.1
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
else:
    raise "ERROR: Please select 0 or 1 as rpu_config"

model = AnalogLSTMModel(rpu_config=rpu_config, device=device, k=args.k)
model = nn.DataParallel(model)
model = model.to(device)

if device == "cuda:0":
    model = nn.DataParallel(model)
    cudnn.benchmark = True

if exists(f'../data/trained_models/{output_file}_k{args.k}_v{args.version}.ckpt'):
    print('Loading model...')
    model.load_state_dict(torch.load(f'../data/trained_models/{output_file}_k{args.k}_v{args.version}.ckpt'))
else:
    print("ERROR: Cannot find trained model at " + f"../data/trained_models/{output_file}_k{args.k}_v{args.version}.ckpt.")
    quit()

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

np.savez_compressed(f"../data/misalign_loss_analog_k{args.k}_v{args.version}.npz", loss=misalign_loss)
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WGANGenerator(nn.Module):
    def __init__(self, noise_dim, time_steps):
        super(WGANGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.time_steps = time_steps

        self.fc1 = nn.Linear(noise_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc_output = nn.ModuleList([nn.Linear(128, 1) for _ in range(time_steps)])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.unsqueeze(1).repeat(1, self.time_steps, 1) 
        x, _ = self.lstm(x)
        x = torch.stack([self.fc_output[i](x[:, i, :]) for i in range(self.time_steps)], dim=1)
        return x



class WGANDiscriminator(nn.Module):
    def __init__(self, time_steps):
        super(WGANDiscriminator, self).__init__()
        self.time_steps = time_steps

        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.ModuleList([nn.Linear(128, 1) for _ in range(time_steps)])

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        x = torch.stack([self.fc[i](lstm_output[:, i, :]) for i in range(self.time_steps)], dim=1)
        return x, lstm_output
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1, 1)).expand_as(real_samples).to(device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = self(interpolates)

        fake = Variable(torch.Tensor(real_samples.size(0), self.time_steps, 1).fill_(1.0), requires_grad=False).to(device)

        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicExpert(nn.Module):
    def __int__(self, feature_in, feature_out):
        self.fc = nn.Linear(feature_in,feature_out)

    def forward(self, x):
        return self.fc(x)

class BasicMOE(nn.Module):
    def __int__(self, feature_in, feature_out, num_expert):
        super().__init__()

        self.gate = nn.Linear(feature_in, num_expert)

        self.experts = nn.ModuleList(
            BasicExpert(
                feature_in, feature_out
            )for _ in range(num_expert)
        )

    def forward(self, X):
        #X shape is (batchsize, hidden_dim=feature_in)
        expert_weight = self.gate(X)
        expert_out_list = [expert(X) for expert in self.experts]

        expert_out

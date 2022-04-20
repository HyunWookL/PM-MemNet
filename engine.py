import torch
import torch.optim as optim
import util
class trainer():
    def __init__(self, scaler, num_nodes, key_dicts, supports, hops, device, seq_length=12, embedding_dim=64, lrate=1e-3, wdecay=0.0, steps = [20,30,40,50]):
        from model.PMMemNet import PMMemNet, init_params
        self.model = PMMemNet(num_nodes, key_dicts, hops, embedding_dim = embedding_dim, sequence_length = seq_length)
        init_params(self.model)
        #self.model.supports(supports)
        self.model.to(device)
        self.supports = supports
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, self.supports, self.scaler.transform(real_val[...,[0]]))
        real = torch.unsqueeze(real_val[...,0],dim=-1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 1e-3)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,1e-3).item()
        rmse = util.masked_rmse(predict,real,1e-3).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input, self.supports)
        real = torch.unsqueeze(real_val[...,0],dim=-1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 1e-3)
        mape = util.masked_mape(predict,real,1e-3).item()
        rmse = util.masked_rmse(predict,real,1e-3).item()
        return loss.item(),mape,rmse

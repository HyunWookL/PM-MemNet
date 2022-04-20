import torch
import numpy as np
import argparse
import time, os
import yaml
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,default='config/PM-MemNet_metr-la.yaml')
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--load_path', type = str, default = None)
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    if args.config is None:
        raise ValueError
    config = yaml.load(open(args.config))
    device = torch.device(args.device)
    print('Start to load dataset...')
    dataloader = util.load_dataset(config['dataset_dir'], config['batch_size'], config['batch_size'], config['batch_size'])
    _, _, adj_mx = util.load_adj(config['adjfile'])
    supports = [torch.Tensor(adj).to(device) for adj in adj_mx]
    print('start to build model...')
    scaler = dataloader['scaler']
    key_dict = np.load(config['prior'])
    key_dict = torch.from_numpy(key_dict).float().view(key_dict.shape[0], -1).to(device)
    num_nodes = config['num_nodes']
    engine = trainer(scaler, num_nodes, key_dict, supports, config['hops'], args.device, seq_length = config['seq_length'], embedding_dim = config['embedding_dim'])
    assert args.load_path is not None
    print("Load Path is detected. Load pretrained model...")
    engine.model.load_state_dict(torch.load(args.load_path))
    print('num of parameters: {}'.format(count_parameters(engine.model)))

    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    s1 = time.time()
    for iter,(x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        with torch.no_grad():
            preds = engine.model(testx, engine.supports)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0),...]
    s2 = time.time()
    print("Inference Time: {:.4f}".format(s2-s1))

    amae = []
    amape = []
    armse = []
    if args.save_output is not None:
        results = {'prediction': [], 'ground_truth':[]}
    else:
        results = None
    from copy import deepcopy as cp
    for i in range(config['seq_length']):
        pred = scaler.inverse_transform(yhat[:,i,:])
        real = realy[:,i,:,[0]]
        if results is not None:
            results['prediction'].append(cp(pred).cpu().numpy())
            results['ground_truth'].append(cp(real).cpu().numpy())
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(config['seq_length'],np.mean(amae),np.mean(amape),np.mean(armse)))
    if args.save_output is not None:
        results['prediction'] = np.asarray(results['prediction'])
        results['ground_truth'] = np.asarray(results['ground_truth'])
        np.savez_compressed(args.save_output, **results)
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))














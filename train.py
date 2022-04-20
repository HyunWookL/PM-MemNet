import torch
import numpy as np
import argparse
import time, os
import yaml
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,default='config/PM-MemNet_metr-la.yaml')
parser.add_argument('--print_every',type=int,default=50)
parser.add_argument('--expid',type=int,default=0)
parser.add_argument('--save',type=str,default='experiment/PM-MemNet', help = 'path and prefix for the model path')
parser.add_argument('--save_output',type=str,default=None, help = 'results and ground truth path. If not given, it will not save the results.')
parser.add_argument('--device',type=int,default=0, help = 'gpu id')
parser.add_argument('--load_path', type = str, default = None, help = 'load path for the training model path should be like ...epoch_XX_....pth')
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
    engine = trainer(scaler, num_nodes, key_dict, supports, config['hops'], args.device, seq_length = config['seq_length'], embedding_dim = config['embedding_dim'],
                     lrate = config['lr'], wdecay = config['wd'], steps = config['steps'])
    start_epoch = 1
    if args.load_path is not None:
        print("Load Path is detected. Load pretrained model...")
        engine.model.load_state_dict(torch.load(args.load_path))
        start_epoch = int(args.load_path.split('epoch_')[-1].split('_')[0]) + 1
    #from torchsummary import summary
    #summary(engine.model, [(207,20), (12,207,9),(12,207,9)])
    print('num of parameters: {}'.format(count_parameters(engine.model)))
    #exit()
    print("start training...",flush=True)
    his_loss = [] if start_epoch == 1 else [1e9 for i in range(start_epoch-1)]
    val_time = []
    train_time = []
    patience = 20
    wait = 0
    for i in range(start_epoch, config['epochs']+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x,y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2-t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            metrics = engine.eval(testx,testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        if np.argmin(his_loss) + 1 == i: wait = 0
        else: wait += 1
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,(t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        if wait >= patience:
            print('Early Termination on Epoch: {:03d}'.format(i))
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+'.pth'))
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
    print("Training finished")
    print("The valid loss on best model: {} is".format(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+'.pth'), str(round(his_loss[bestid],4)))
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














import os
import time
import socket
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from DORN_Network import DORN
from load_data import getNYUDataset, get_depth_sid

init_lr = 0.0001
momentum = 0.9
epoches = 140
batch_size = 2
max_iter = 9000000
resume = True # Whether the model is saved
model_path = '.\\run\\checkpoint-119.pth.tar'
output_dir = '.\\run'



class ordLoss(nn.Module):
    def __init__(self):
        super(ordLoss, self).__init__()
        self.loss = 0.0
    def forward(self, ord_labels, target):
        N, C, H, W = ord_labels.size()
        ord_num = C
        if torch.cuda.is_available():
            K = torch.zeros((N, C, H, W), dtype=torch.float32).cuda()
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.float32).cuda()
        else:
            K = torch.zeros((N, C, H, W), dtype=torch.float32)
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.float32)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size())
        if torch.cuda.is_available():
            one = one.cuda()

        self.loss = torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-7, max=1e7))) \
                    + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-7, max=1e7)))

        N = N * H * W
        self.loss /= (-N)
        return self.loss



def update_ploy_lr(optimizer, initialized_lr, current_step, max_step, power=0.9):
    lr = initialized_lr * ((1 - float(current_step) / max_step) ** (power))
    idx = 0
    for param_group in optimizer.param_groups:
        if idx == 0: # base params
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * 20
        idx += 1
    return lr

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def main():
    train_loader, val_loader, test_loader = getNYUDataset()
    best_result = Result()
    best_result.set_to_worst()

    if resume:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model_dict = checkpoint['model']
        model = DORN()
        model.load_state_dict(model_dict)
        model = checkpoint['model']


        # aspp module's lr is 20 bigger than the other modules in paper
        aspp_params = list(map(id, model.aspp_module.parameters()))
        base_params = filter(lambda p: id(p) not in aspp_params, model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        print("loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  
        del model_dict


    else:

        model = DORN()
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum)
        start_epoch = 0

    if torch.cuda.device_count():
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    criterion = ordLoss()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_txt = os.path.join(output_dir, 'best.txt')
    log_path = os.path.join(output_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)



    # Begin training
    for  epoch in range(start_epoch, epoches):
        train(train_loader, model, criterion, optimizer, epoch, logger)
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nrmse={:.3f}\nrml={:.3f}\nlog10={:.3f}\nd1={:.3f}\nd2={:.3f}\ndd31={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_dir + '/comparison_best.png'
                save_image(img_merge, img_filename)

        save_checkpoint({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'best_result': best_result},
                        is_best, epoch, output_dir)



# One epoch in NYU dataset
def train(train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()
    end = time.time()
    batch_num = len(train_loader)
    current_step = batch_num * batch_size * epoch
    for i, (input, target) in enumerate(train_loader):
        lr = update_ploy_lr(optimizer, init_lr, current_step, max_iter)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        current_step += input.data.shape[0]

        if current_step == max_iter:
            logger.close()
            break
        torch.cuda.synchronize()

        end = time.time()
        # compute pred
        end = time.time()
        with torch.autograd.detect_anomaly():
            pred_d, pred_ord = model(input)  

            loss = criterion(pred_ord, target)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

        torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        depth = get_depth_sid(pred_d)
        target_dp = get_depth_sid(target)
        result.evaluate(depth.data, target_dp.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'learning_rate={lr:.8f} '
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={loss:.3f} '
                  'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
                  'RML={result.absrel:.3f}({average.absrel:.3f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
                gpu_time=gpu_time, result=result, average=average_meter.average()))

            logger.add_scalar('Learning_rate', lr, current_step)
            logger.add_scalar('Train/Loss', loss.item(), current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)
        avg = average_meter.average()

class AverageMeter(object):
def __init__(self):
    self.reset()

def reset(self):
    self.count = 0.0

    self.sum_irmse, self.sum_imae = 0, 0
    self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
    self.sum_absrel, self.sum_lg10 = 0, 0
    self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
    self.sum_data_time, self.sum_gpu_time = 0, 0

def update(self, result, gpu_time, data_time, n=1):
    self.count += n

    self.sum_irmse += n*result.irmse
    self.sum_imae += n*result.imae
    self.sum_mse += n*result.mse
    self.sum_rmse += n*result.rmse
    self.sum_mae += n*result.mae
    self.sum_absrel += n*result.absrel
    self.sum_lg10 += n*result.lg10
    self.sum_delta1 += n*result.delta1
    self.sum_delta2 += n*result.delta2
    self.sum_delta3 += n*result.delta3
    self.sum_data_time += n*data_time
    self.sum_gpu_time += n*gpu_time

def average(self):
    avg = Result()
    avg.update(
        self.sum_irmse / self.count, self.sum_imae / self.count,
        self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
        self.sum_absrel / self.count, self.sum_lg10 / self.count,
        self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
        self.sum_gpu_time / self.count, self.sum_data_time / self.count)
    return avg


def log10(x):
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

if __name__ == '__main__':
    main()

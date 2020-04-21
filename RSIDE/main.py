import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import network
import resnet
import edge_accuracy
import time
import data
import math
import numpy as np


def train(train_loader, model, optimizer, epoch):

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = edge_accuracy.Sobel()

    criterion = nn.L1Loss()
    batch_time = Notes()
    losses = Notes()

    model.train()

    end = time.time()

    for i, sample_batch in enumerate(train_loader):

        image, depth = sample_batch['image'], sample_batch['depth']
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)

        optimizer.zero_grad()

        output = model(image)

        ## Loss function

        depth_gradient = get_gradient(depth)
        depth_gradient_x = depth_gradient[:, 0, :, :].contiguous().view_as(depth)
        depth_gradient_y = depth_gradient[:, 1, :, :].contiguous().view_as(depth)

        estimated_gradient = get_gradient(output)
        estimated_gradient_x = estimated_gradient[:, 0, :, :].contiguous().view_as(depth)
        estimated_gradient_y = estimated_gradient[:, 1, :, :].contiguous().view_as(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float()
        ones = torch.autograd.Variable(ones)

        l_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

        l_dx = torch.log(torch.abs(estimated_gradient_x - depth_gradient_x) + 0.5).mean()
        l_dy = torch.log(torch.abs(estimated_gradient_y - depth_gradient_y) + 0.5).mean()
        l_grad = l_dx + l_dy

        depth_normal = torch.cat((-depth_gradient_x, -depth_gradient_y, ones), 1)
        estimated_normal = torch.cat((-estimated_gradient_x, -estimated_gradient_y, ones), 1)
        l_normal = torch.abs(1 - cos(estimated_normal, depth_normal)).mean()

        loss = l_depth + l_grad + l_normal

        losses.update(loss.data, image.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end)
        end = time.time()

        batch_size = depth.size(0)

        print(epoch, i, len(train_loader), batch_time.values, batch_time.sum, losses.values, losses.average)
        print(batch_size)



def update_lr(optimizer, epoch, lr=0.0001):

    lr = lr * (0.1 ** (epoch // 5))

    for p in optimizer.param_groups:
        p['lr'] = lr



class Notes(object):

    def __init__(self):
        self.reset()        


    def reset(self):
        self.values = 0
        self.sum = 0
        self.count = 0
        self.average = 0


    def update(self, value, n=1):
        self.values = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count



def maxTwo(x, y):

    z = x.clone()
    y_larger = torch.lt(x, y)
    z[y_larger.detach()] = y[y_larger.detach()]
    return z



def set_nan_to_zero(inputs, targets):

    mask = torch.ne(targets, targets)
    n_valid_elements = torch.sum(torch.eq(targets, targets).float())

    _inputs = inputs.clone()
    _targets = targets.clone()

    _inputs[mask] = 0
    _targets[mask] = 0

    return _inputs, _targets, mask, n_valid_elements



def evaluate_errors(outputs, targets):

    errors = {
        'MSE': 0,
        'RMSE': 0,
        'ABS_REL': 0,
        'LG10': 0,
        'MAE': 0,
        'DELTA1': 0,
        'DELTA2': 0,
        'DELTA3': 0
    }

    _outputs, _targets, mask, n_valid_elements = set_nan_to_zero(outputs, targets)

    if (n_valid_elements.data.cpu().numpy() > 0):

        diff_matrix = torch.abs(_outputs - _targets)

        errors['MSE'] = torch.sum(torch.pow(diff_matrix, 2)) / n_valid_elements
        errors['MAE'] = torch.sum(diff_matrix) / n_valid_elements

        r_matrix = torch.div(diff_matrix, _targets)
        r_matrix[mask] = 0
        errors['ABS_REL'] = torch.sum(r_matrix) / n_valid_elements

        lg10_matrix = torch.abs(torch.div(torch.log(_outputs), math.log(10)) - torch.div(torch.log(_targets), math.log(10)))
        lg10_matrix[mask] = 0
        errors['LG10'] = torch.sum(lg10_matrix) / n_valid_elements

        max_ratio = maxTwo(torch.div(_outputs, _targets), torch.div(_targets, _outputs))

        errors['DELTA1'] = torch.sum(torch.le(max_ratio, 1.25).float()) / n_valid_elements
        errors['DELTA2'] = torch.sum(torch.le(max_ratio, math.pow(1.25, 2)).float()) / n_valid_elements
        errors['DELTA3'] = torch.sum(torch.le(max_ratio, math.pow(1.25, 3)).float()) / n_valid_elements

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

        return errors



def add_errors(error_sum, errors, batch_size):

    error_sum['MSE'] = error_sum['MSE'] + errors['MSE'] * batch_size
    error_sum['ABS_REL'] = error_sum['ABS_REL'] + errors['ABS_REL'] * batch_size
    error_sum['LG10'] = error_sum['LG10'] + errors['LG10'] * batch_size
    error_sum['MAE'] = error_sum['MAE'] + errors['MAE'] * batch_size
    error_sum['DELTA1'] = error_sum['DELTA1'] + errors['DELTA1'] * batch_size
    error_sum['DELTA2'] = error_sum['DELTA2'] + errors['DELTA2'] * batch_size
    error_sum['DELTA3'] = error_sum['DELTA3'] + errors['DELTA3'] * batch_size

    return error_sum



def avg_errors(error_sum, n):

    avg_errs = {
        'MSE': 0,
        'RMSE': 0,
        'ABS_REL': 0,
        'LG10': 0,
        'MAE': 0,
        'DELTA1': 0,
        'DELTA2': 0,
        'DELTA3': 0
    }

    avg_errs['MSE'] = error_sum['MSE'] / n
    avg_errs['ABS_REL'] = error_sum['ABS_REL'] / n
    avg_errs['LG10'] = error_sum['LG10'] / n
    avg_errs['MAE'] = error_sum['MAE'] / n
    avg_errs['DELTA1'] = error_sum['DELTA1'] / n
    avg_errs['DELTA2'] = error_sum['DELTA2'] / n
    avg_errs['DELTA3'] = error_sum['DELTA3'] / n

    return avg_errs



def test(test_loader, model, t):

    model.eval()

    error_sum = {
        'MSE': 0,
        'RMSE': 0,
        'ABS_REL': 0,
        'LG10': 0,
        'MAE': 0,
        'DELTA1': 0,
        'DELTA2': 0,
        'DELTA3': 0
    }

    total_number = 0

    with torch.no_grad():

        for i, sample_batch in enumerate(test_loader):

            print(i)

            image, depth = sample_batch['image'], sample_batch['depth']

            image = torch.autograd.Variable(image)
            depth = torch.autograd.Variable(depth)

            output = model(image)
            output = torch.nn.functional.interpolate(output, size=[depth.size(2), depth.size(3)], mode='bilinear')

            batch_size = depth.size(0)
            total_number = total_number + batch_size

            errors = evaluate_errors(output, depth)
            error_sum = add_errors(error_sum, errors, batch_size)
            avg_errs = avg_errors(error_sum, total_number)


    avg_errs['RMSE'] = np.sqrt(avg_errs['MSE'])
    print(avg_errs)


def main():

    model = network.Model(resnet.resnet50(pretrained=True), num_features=2048, input_channels=[256,512,1024,2048])
    batch_size = 8
    epochs = 5
    lr = 0.0001
    wd = 1e-4

    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), 0.0001, weight_decay=1e-4)

    
    ## training

    train_file_url = './data/nyu2_train.csv'
    train_loader = data.training_data(train_file_url, batch_size)

    for epoch in range(0, epochs):
        update_lr(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)    
    
    ## testing

    test_file_url = './data/nyu2_test.csv'
    test_loader = data.testing_data(test_file_url, 1)

    test(test_loader, model, 0.25)

    


if __name__ == "__main__":
    main()
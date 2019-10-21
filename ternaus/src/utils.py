import json
import random
import shutil
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize, Compose

print('reading input config file...')
config = json.loads(open(str(Path('__file__').absolute().parent.parent / 'config' / 'config.json')).read())

DATA_ROOT = Path(config['input_data_dir']).expanduser()
SUBMISSION_PATH = Path(config['submissions_dir']).expanduser()
MODEL_PATH = Path(config['models_dir']).expanduser()

cuda_is_available = torch.cuda.is_available()

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def variable(x, volatile=False):
    '''
    Move variables to GPU.
    :param x: input variable
    :param volatile: Depreciated in PyTorch v 0.4
    :return:
    '''
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if cuda_is_available else x


def write_event(log, step: int, **data):
    '''
    Write event to log file
    :param log:
    :param step:
    :param data:
    :return:
    '''
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def add_args(parser):
    arg = parser.add_argument
    arg('--root', default='models/unet_11', help='checkpoint root')
    arg('--batch-size', type=int, default=24)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)


def cyclic_lr(epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    '''
    Implements a cyclical exponential-decaying learning rate. init_lr is the maximum learning reate. See Notebook in Analysis dir.

    :param epoch: current epoch
    :param init_lr:
    :param num_epochs_per_cycle:
    :param cycle_epochs_decay:
    :param lr_decay_factor:
    :return:
    '''

    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))

    return lr


def train(args, model: nn.Module, criterion, *, train_loader, valid_loader,
          validation, init_optimizer, fold=None, save_predictions=None, n_epochs=None):
    '''

    :param args:
    :param model:
    :param criterion:
    :param train_loader:
    :param valid_loader:
    :param validation:
    :param init_optimizer:
    :param fold:
    :param save_predictions:
    :param n_epochs:
    :return:
    '''

    n_epochs = n_epochs or args.n_epochs

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best-model_{fold}.pt'.format(fold=fold)

    # if model checkpoints exist, load the latest
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    # saving function for model
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20

    # this logging is not necessary with MLflow
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')

    valid_losses = []

    for epoch in range(epoch, n_epochs + 1):

        # cyclical learning rate
        lr = cyclic_lr(epoch)

        # Adam is re-initialized in each epoch, with the cyclical learning rate value as input.
        optimizer = init_optimizer(lr)

        model.train()

        # why random seed here?
        random.seed()

        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))

        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))

        losses = []

        tl = train_loader
        if args.epoch_size:
            # move start of traing loader tl by the number of steps
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):

                # move inputs to GPU
                inputs, targets = variable(inputs), variable(targets)

                # get model output
                outputs = model(inputs)

                # calculate loss
                loss = criterion(outputs, targets)

                optimizer.zero_grad()

                batch_size = inputs.size(0)
                step += 1
                tq.update(batch_size)
                #print('loss test ',loss.item())
                #losses.append(loss.data[0]) CR changed
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])

                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

                #???
                (batch_size * loss).backward()

                optimizer.step()

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)

            write_event(log, step, loss=mean_loss)
            tq.close()

            # save model
            save(epoch + 1)

            # returns avg loss and and dice loss
            valid_metrics = validation(model, criterion, valid_loader)

            # los results
            write_event(log, step, **valid_metrics)

            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            # if best validation loss copy model to best model location.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


# being used??
def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

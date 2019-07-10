import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger

import models.anynet

if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')

parser.add_argument("--test", action="store_true", default=False, \
    help="Issue this argument for testing a trained mode only. This option must be used with --resume option.")

parser.add_argument("--test_stride", type=int, default=100, \
    help="The stride size for indexing the testing input images.")

args = parser.parse_args()


def main():
    global args

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    if ( args.test ):
        logFn = "/testing.log"
        if ( not os.path.isdir( args.save_path + "/Testing" ) ):
            os.makedirs( args.save_path + "/Testing" )

        print("\n=== Testing ===")
    else:
        logFn = "/training.log"

    log = logger.setup_logger(args.save_path + logFn)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    if args.test:
        # Only perform test. --resume option is assumed to be issued at the same time.
        if ( args.resume is None ):
            raise Exception("--resume arguments must be set while --test is issued.")
        
        if ( not os.path.isfile( args.save_path + "/" + args.resume ) ):
            raise Exception("Checkpoint %s does not exist." % ( args.save_path + "/" + args.resume ))

        log.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load( args.save_path + "/" + args.resume )
        model.load_state_dict(checkpoint['state_dict'])
    elif args.resume:
        if os.path.isfile( args.save_path + "/" + args.resume ):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load( args.save_path + "/" + args.resume )
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()
    
    if ( not args.test ):
        for epoch in range(args.start_epoch, args.epochs):
            log.info('This is {}-th epoch'.format(epoch))

            train(TrainImgLoader, model, optimizer, log, epoch)

            savefilename = args.save_path + '/checkpoint.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)

    if ( not args.test ):
        test(TestImgLoader, model, log)
    else:
        test( TestImgLoader, model, log, args.test_stride, args.save_path + "/Testing" )

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L < args.maxdisp
        mask.detach_()
        outputs = model(imgL, imgR)
        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

def save_test_results(savePath, batchIdx, imgL, imgR, dispL, dispPred):
    """
    Save tsting results to savePath.
    imgL, imgR, dispL, and dispPred are all numpy tensors, with their first dimension 
    representing the batch size. The actual dimensions of these input arguments are 
    assumed to be as follows:
    imgL( B, C, H, W )
    imgR( B, C, H, W )
    dispL( B, H, W )
    dispPred( B, H, W )

    All these tensors are already transfered to cpu.
    
    The filenames for the saved images are composed as "batchIdx_index in minibatch.png".
    """

    for i in range( imgL.shape[0] ):
        iL = np.transpose( imgL[i, :, :, :], (1, 2, 0) )
        iR = np.transpose( imgR[i, :, :, :], (1, 2, 0) )
        dL = dispL[i, :, :]
        dP = dispPred[i, :, :]

        gdtMin = dL.min()
        gdtMax = dL.max()

        dP = dP - gdtMin
        dL = dL - gdtMin

        dP = np.clip( dP / gdtMax, 0.0, 1.0 )
        dL = dL / gdtMax

        fig = plt.figure( figsize=(12.8, 9.6), dpi=300 )

        ax = plt.subplot(2, 2, 1)
        plt.tight_layout()
        ax.set_title("Ref")
        ax.axis("off")
        iL = iL - iL.min()
        iL = iL / iL.max()
        plt.imshow( iL )

        ax = plt.subplot(2, 2, 3)
        plt.tight_layout()
        ax.set_title("Tst")
        ax.axis("off")
        iR = iR - iR.min()
        iR = iR / iR.max()
        plt.imshow( iR )

        ax = plt.subplot(2, 2, 2)
        plt.tight_layout()
        ax.set_title("Ground truth")
        ax.axis("off")
        plt.imshow( dL )

        ax = plt.subplot(2, 2, 4)
        plt.tight_layout()
        ax.set_title("Prediction")
        ax.axis("off")
        plt.imshow( dP )

        figName = "%s/%s_%02d.png" % (savePath, batchIdx, i)
        plt.savefig(figName)

        plt.close(fig)

def test(dataloader, model, log, stride=1, savePath=None):

    stages = 3 + args.with_spn
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        if ( batch_idx % stride != 0 ):
            continue
        
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                output = output[:, 4:, :]
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        if ( savePath is not None ):
            # Save the images.
            save_test_results(savePath, batch_idx, \
                imgL.cpu().numpy()[:, :, 4:, :], imgR.cpu().numpy()[:, :, 4:, :], disp_L.cpu().numpy(), output.cpu().numpy())
            # # For debugging use.
            # break

        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])

        log.info('[T {}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

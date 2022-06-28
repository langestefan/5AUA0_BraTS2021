import argparse
from posixpath import split
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import os
import matplotlib
import matplotlib.pyplot as plt  # we need to import this to make use of matplotlib.image.imsave
import numpy as np

def save_image(seg_img, filename):
    color_segmentation = np.zeros((240, 240, 3), dtype=np.uint8)

    # map each class to a specific color in [black, red, green, blue]
    color_segmentation[seg_img[:, :] == 0] = [0, 0, 0]      # Black (healthy tissue) = 0
    color_segmentation[seg_img[:, :] == 1] = [255, 0, 0]    # Red (necrotic tumor core) = 1
    color_segmentation[seg_img[:, :] == 2] = [0, 255,0]     # Green (peritumoral edematous/invaded tissue) = 2
    color_segmentation[seg_img[:, :] == 3] = [0, 0, 255]    # Blue (enhancing tumor) = 4

    # store the result as RGB PNG
    matplotlib.image.imsave(filename, color_segmentation)


def main(config):
    logger = config.get_logger('test')
    print("batch_size: ", config['data_loader']['args']['batch_size'])

    # test batch size can be larger due to no backpropagation
    batch_size_test = 3 * config['data_loader']['args']['batch_size']
    print("batch_size_test: ", batch_size_test)

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=batch_size_test,
        shuffle=False,
        validation_split=0.0,
        split='test',
        num_workers=1,

        # for testing we always take all the slices
        start_slice=0, 
        end_slice=154        
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0

    # create empty list length metric_fns
    total_metrics = [[] for i in range(len(metric_fns))]    

    # test dataset properties
    # Get the list of all the files in the directory
    files_per_case = 755 # 155 slices per case, 4 modalities + 1 for seg = 155*5

    n_cases = data_loader.dataset.n_cases
    cases = data_loader.dataset.cases     

    # there are 155 slices per case, 4 modalities + 1 seg mask --> 155 * 5 = 775 files per case 
    start_slice = data_loader.dataset.start_slice
    end_slice = data_loader.dataset.end_slice  
    slices_per_case = data_loader.dataset.slices_per_case 

    print("Number of cases: ", n_cases)
    print("Number of slices per case: ", slices_per_case)
    print("Start slice: ", start_slice)
    print("End slice: ", end_slice)
    print("Number of files per case: ", files_per_case)

    # to store all outputs
    all_pred_segs = []
    all_target_segs = []


    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            n_samples = len(data)

            # save output as images
            if config['store_output']['save_images']:
                save_dir = os.path.join(config['store_output']['image_dir'], config['name'])

                target_dir = os.path.join(save_dir, 'target')
                pred_dir = os.path.join(save_dir, 'pred')

                if not os.path.exists(target_dir):
                    print("Creating directory: ", target_dir)
                    os.makedirs(target_dir)     

                if not os.path.exists(pred_dir):               
                    print("Creating directory: ", pred_dir)
                    os.makedirs(pred_dir)

                # loop over the current batch
                for j in range(n_samples):
                    
                    # img index
                    index = i * batch_size_test + j

                    # convert index to case lookup
                    case_idx = index // slices_per_case
                    slice_idx = index % slices_per_case + start_slice # add start_slice because slices are in range (start_slice, end_slice)

                    # assert that the index is in range (start_slice, end_slice) and case is in range (0, n_cases) 
                    assert slice_idx >= start_slice and slice_idx <= end_slice, 'Index (%d) out of range' % slice_idx
                    assert case_idx >= 0 and case_idx < n_cases, 'Case index (%d) out of range' % case_idx                 

                    # get case name
                    case_name = cases[case_idx]
                    path_out = os.path.join(save_dir, 'pred', f'{case_name}_{slice_idx}_seg_pred.png')

                    ### save output as image ###
                    seg_out_img = output[j].cpu()
                    seg_target_img = target[j].cpu()

                    # store output and target 
                    all_pred_segs.append(seg_out_img)
                    all_target_segs.append(seg_target_img)

                    # the class for a given pixel is the class with the highest probability
                    seg_img = seg_out_img
                    seg_img = torch.argmax(seg_img, dim=0)
                    seg_img = torch.transpose(seg_img, 0, 1)
                    # print(f"Classes predicted: {np.unique(seg_img.numpy())}")
                    save_image(seg_img, path_out)

                    ### save ground truth as image ###
                    seg_img_tar = seg_target_img
                    seg_img_tar = torch.argmax(seg_img_tar, dim=0)
                    seg_img_tar = torch.transpose(seg_img_tar, 0, 1)
                    path_tar = os.path.join(save_dir, 'target', f'{case_name}_{slice_idx}_seg_target.png')
                    save_image(seg_img_tar, path_tar)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            # compute metrics on test set if we have a full case (155 slices)
            if len(all_pred_segs) >= 155:
                # get the current case
                len_all_pred_segs = len(all_pred_segs)
                len_all_target_segs = len(all_target_segs)
                case_pred = all_pred_segs[0:155]
                case_target = all_target_segs[0:155]

                # and then remove it from the list
                del all_pred_segs[0:155]
                del all_target_segs[0:155]

                # make sure we actually removed the right number of slices
                assert len(all_pred_segs) == len(all_target_segs), 'Number of slices in pred and target do not match'
                assert len_all_pred_segs - 155 == len(all_pred_segs), 'Did not remove all slices from pred' 

                # cat the current case
                case_pred = torch.stack(case_pred, dim=0)
                case_target = torch.stack(case_target, dim=0)

                # metrics dict
                metrics = {0: "WT", 1: "TC", 2: "ET"}

                # compute metrics for the current case       
                for i, metric in enumerate(metric_fns):
                    metric_out = metric(case_pred, case_target)

                    assert metric_out.numel() > 0 and not torch.isnan(metric_out).any(), 'Metric output is NaN or empty'

                    # print case name + metric
                    logger.info(f"{case_name}: {metrics[i]}={metric_out.item():.4f}")

                    # store metric output
                    total_metrics[i].append(metric_out.item())



    n_samples = len(data_loader.sampler)

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: sum(total_metrics[i]) / len(total_metrics[i])  for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    # python test.py -r saved/models/BraTS2021_Base_Unet/0528_015638/model_best.pth
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

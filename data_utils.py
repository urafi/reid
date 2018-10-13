import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import time
import torchvision
from tqdm import tqdm

def get_transform(param, crop_pos, output_size, scales):

    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    a = scales[0] * param['scale'] * np.cos(param['rot'])
    b = scales[1] * param['scale'] * np.sin(param['rot'])

    t = np.identity(3)
    t[0][0] = a
    if param['flip']:
        t[0][0] = -a

    t[0][1] = -b
    t[1][0] = b
    t[1][1] = a

    shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
    shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
    shift_to_center[0][2] = output_size / 2
    shift_to_center[1][2] = output_size / 2
    t_form = np.matmul(t, shift_to_upper_left)
    t_form = np.matmul(shift_to_center, t_form)

    return t_form


def train(train_loader, model, optimiser, criterion):

    # switch to train mode
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (img, ids) in enumerate(tqdm(train_loader)):

        ids = ids.cuda(non_blocking=True)
        inputs = img.cuda(non_blocking=True)

        output = model(inputs)

        l = criterion(output, ids)

        total_loss += l.data.item()

        optimiser.zero_grad()
        l.backward()
        optimiser.step()

        #print ('TrainEpoch [%d], Iter [%d/%d] , loss [%f]'
        #       % (epoch + 1, i + 1, NBatches, l.data.item()))

    print('Total Time for train epoch %.4f' % (time.time() - start_time))

    return total_loss


def test(val_loader, model, epoch):

    model.eval()
    det = []

    with torch.no_grad():

        for i, sampled_batch in enumerate(tqdm(val_loader)):

            images = sampled_batch['images']
            imagesf = sampled_batch['imagesf']
            inputs = images.cuda(non_blocking=True)
            inputsf = imagesf.cuda(non_blocking=True)

            single_result_dict = {}

            output = model(inputs)

            sr = output[:, 17:51, :, :].data.cpu()

            output = torch.sigmoid(output[:, 0:17])
            output = output.data.cpu()

            outputf = model(inputsf)
            outputf = torch.sigmoid(outputf[:, 0:17])
            outputf = outputf.data.cpu()


            # sr = output[:, 17:51, :, :].data.cpu()
            # lr = output[:, 51:115, :, :].data.cpu()
            N = output.shape[0]

            for n in range(N):

                single_result_dict = {}
                prs = torch.zeros(17, 64, 64)
                outputflip = outputf[n]
                outputflip = outputflip[flipRef]
                for j in range(17):
                    prs[j] = output[n][j] + torch.from_numpy(cv2.flip(outputflip[j].numpy(), 1))

                keypoints, score = get_preds(prs, sampled_batch['meta']['warps'][n], sr[n])

                single_result_dict['image_id'] = int(sampled_batch['meta']['imgID'][n].item())
                single_result_dict['category_id'] = 1
                single_result_dict['keypoints'] = keypoints
                single_result_dict['score'] = score

                det.append(single_result_dict)

    with open('dt.json', 'w') as f:
        json.dump(det, f)

    eval_gt = COCO('../Coco/annotations/person_keypoints_val2017.json')
    eval_dt = eval_gt.loadRes('dt.json')
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[0]

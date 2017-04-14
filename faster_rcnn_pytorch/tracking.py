import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import os
import ipdb
import h5py
from Dataloader import DAVIS_seq_dataloader
from UNet_model import UNet

import torch
import torch.nn.functional as F
from torch.autograd import Variable


_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# A helper function to get bbox from segmentation
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    height = rmax - rmin + 1
    width = cmax - cmin + 1
    h_delta = 0.1 * height
    w_delta = 0.1 * width

    rmin = np.maximum(0, rmin - h_delta)
    rmax = np.minimum(img.shape[0], rmax + h_delta)
    cmin = np.maximum(0, cmin - w_delta)
    cmax = np.minimum(img.shape[1], cmax + w_delta)

    return rmin, rmax, cmin, cmax

def merge_rois(bbox, rois, thres=0.75):
    rmin, rmax, cmin, cmax = bbox
    bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1) 

    num_rois = rois.shape[0]
    x1 = rois[:, 1]
    y1 = rois[:, 2]
    x2 = rois[:, 3]
    y2 = rois[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    for i in range(num_rois):
        xx1 = np.maximum(cmin, x1[i])
        yy1 = np.maximum(rmin, y1[i])
        xx2 = np.minimum(cmax, x2[i])
        yy2 = np.minimum(rmax, y2[i])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + bbox_area - inter)
        if ovr >= thres:
            keep.append(i)

    if len(keep) == 0:
        return None, None, None, None

    new_rois = rois[keep]
    x1 = np.min(new_rois[:,1])
    y1 = np.min(new_rois[:,2])
    x2 = np.max(new_rois[:,3])
    y2 = np.max(new_rois[:,4])
    return x1, y1, x2, y2

def test():
    # Set up dataloader
    data_loader = DAVIS_seq_dataloader(split='val')

    model_file = './model/VGGnet_fast_rcnn_iter_70000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('Load Faster R-CNN model successfully!')

    # unet_model = './model/vgg_unet_1e-4_500.h5'
    # unet = UNet()
    # network.load_net(unet_model, unet)
    # unet.cuda()
    # network.weights_normal_init(unet, dev=0.01)
    # unet.load_from_faster_rcnn_h5(h5py.File(model_file))
    criterion_bce = torch.nn.BCELoss().cuda()
    weight_decay = 5e-5
    # optimizer = torch.optim.SGD(list(unet.parameters())[26:], lr=1e-4, weight_decay=weight_decay)
    # print('Load U-Net model successfully!')

    crop_set = []
    # Iterate
    for i in range(data_loader.num_seq):
        # Get the first frame info
        seq = data_loader.seq_list[data_loader.out_pointer]
        seq_len = data_loader.seq_len[seq]
        img_blobs, seg_blobs = data_loader.get_next_minibatch()
        img = img_blobs[0,:,:,:]
        im_data, im_scales = detector.get_image_blob(img)
        im_info = np.array([[im_data.shape[1], im_data.shape[2], im_scales[0]]], dtype=np.float32)
        # Get the category of the object in the first frame
        rmin, rmax, cmin, cmax = bbox(seg_blobs[0,:,:,0])
        features, rois = detector(im_data, im_info, rpn_only=True)
        new_rois_np = np.array([[0, cmin, rmin, cmax, rmax]], dtype=np.float32)
        new_rois_t = torch.from_numpy(new_rois_np).cuda()
        new_rois = Variable(new_rois_t, requires_grad=False)
        pooled_features = detector.roi_pool(features, new_rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = detector.fc6(x)
        x = detector.fc7(x)
        cls_score = detector.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = detector.bbox_fc(x)
        cls_prob_np = cls_prob.cpu().data.numpy()
        bbox_pred_np = bbox_pred.cpu().data.numpy()
        cls_idx = cls_prob_np.argmax()
        cls_conf = cls_prob_np.max()

        # Overfit U-Net with the first frame
        # for i in range(100):
        #     unet.train()
        #     img_t = torch.from_numpy(img_blobs).permute(0,3,1,2).float().cuda()
        #     img_v = Variable(img_t, requires_grad=False)
        #     seg_t = torch.from_numpy(seg_blobs).permute(0,3,1,2).float().cuda()
        #     seg_v = Variable(seg_t, requires_grad=False)
        #     pred = unet(img_v)
            # loss = criterion_bce(pred, seg_v)
        #     pred_view = pred.view(-1, 1)
        #     seg_view = seg_v.view(-1, 1)    
        #     EPS = 1e-6
        #     loss = 0.6 * seg_view.mul(torch.log(pred_view+EPS)) + 0.4 * seg_view.mul(-1).add(1).mul(torch.log(1-pred+EPS))
        #     loss = -torch.mean(loss)
        #     loss_val = loss.data[0]
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print('{}/100: {}'.format(i, loss_val))
        # unet.eval()

        # Merge region proposals overlapping with last frame proposal
        for j in range(1, seq_len):
            img_blobs, _ = data_loader.get_next_minibatch()
            img = img_blobs[0,:,:,:]
            im_data, im_scales = detector.get_image_blob(img)
            # 300 x 5, the first elements are useless here
            features, rois = detector(im_data, im_info, rpn_only=True)
            x1, y1, x2, y2 = merge_rois((rmin, rmax, cmin, cmax), rois.cpu().data.numpy(), thres=0.75)

            # Have overlapping proposals
            if x1 is not None:
                # Send to following layers to refine the bbox
                new_rois_np = np.array([[0, x1, y1, x2, y2]], dtype=np.float32)
                new_rois_t = torch.from_numpy(new_rois_np).cuda()
                new_rois = Variable(new_rois_t, requires_grad=False)
                pooled_features = detector.roi_pool(features, new_rois)
                x = pooled_features.view(pooled_features.size()[0], -1)
                x = detector.fc6(x)
                x = detector.fc7(x)
                cls_score = detector.score_fc(x)
                cls_prob = F.softmax(cls_score)
                bbox_pred = detector.bbox_fc(x)
                cls_prob_np = cls_prob.cpu().data.numpy()
                bbox_pred_np = bbox_pred.cpu().data.numpy()

                # Only regress bbox when confidence is greater than 0.8
                if cls_prob_np.max() > 0.8 and cls_prob_np.argmax() != 0:
                    keep = cls_prob_np.argmax()
                    pred_boxes, scores, classes = detector.interpret_faster_rcnn(cls_prob, bbox_pred, new_rois, im_info, im_data.shape, 0.8)

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    width = x2 - x1 + 1
                    height = y2 - y1 + 1
                    dx = bbox_pred_np[0,keep*4+0]
                    dy = bbox_pred_np[0,keep*4+1]
                    dw = bbox_pred_np[0,keep*4+2]
                    dh = bbox_pred_np[0,keep*4+3]
            
                    pred_x = dx * width + cx
                    pred_y = dy * height + cy
                    pred_w = np.exp(dw) * width
                    pred_h = np.exp(dh) * height

                    x1 = pred_x - pred_w / 2
                    x2 = pred_x + pred_w / 2
                    y1 = pred_y - pred_h / 2
                    y2 = pred_y + pred_h / 2

            # No overlapping proposals
            if x1 is None:
                # Using Faster R-CNN again to find potential objects
                dets, scores, classes = detector.detect(img, 0.6)
                # Cannot find any salient object
                if dets.shape[0] == 0:
                    x1, y1, x2, y2 = cmin, rmin, cmax, rmax
                else:
                    x1 = dets[:,0]
                    y1 = dets[:,1]
                    x2 = dets[:,2]
                    y2 = dets[:,3]
                    pred_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    init_area = (cmax - cmin + 1) * (rmax - rmin + 1)
                    xx1 = np.maximum(x1, cmin)
                    xx2 = np.minimum(x2, cmax)
                    yy1 = np.maximum(y1, rmin)
                    yy2 = np.minimum(y2, rmax)
                    inter = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
                    ovr = inter / (pred_area + init_area - inter)
                    # If there is overlapping, choose the largest IoU bbox
                    try:
                        ovr = ovr[ovr > 0.3]
                        ovr_idx = np.argsort(ovr)[-1]
                        x1 = dets[ovr_idx,0]
                        y1 = dets[ovr_idx,1]
                        x2 = dets[ovr_idx,2]
                        y2 = dets[ovr_idx,3]
                    # Else, choose the highest objectness score one
                    except:
                        if cls_idx == 0:
                            temp_idx = scores.argmax()
                            x1 = dets[temp_idx,0]
                            y1 = dets[temp_idx,1]
                            x2 = dets[temp_idx,2]
                            y2 = dets[temp_idx,3]
                        else:
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            cc = (cmin + cmax) / 2
                            cr = (rmin + rmax) / 2
                            dist = np.sqrt(np.square(cx-cc) + np.square(cy-cr))
                            dist_idx = np.argsort(dist)
                            for di in dist_idx:
                                if classes[di] == _CLASSES[cls_idx]:
                                    x1 = dets[di,0]
                                    y1 = dets[di,1]
                                    x2 = dets[di,2]
                                    y2 = dets[di,3]

            # Crop the region and send it to U-Net
            try:
                x1 = int(max(x1, 0))
                x2 = int(min(x2, im_data.shape[2]))
                y1 = int(max(y1, 0))
                y2 = int(min(y2, im_data.shape[1]))
            except:
                x1 = int(max(x1[0], 0))
                x2 = int(min(x2[0], im_data.shape[2]))
                y1 = int(max(y1[0], 0))
                y2 = int(min(y2[0], im_data.shape[1]))

            # MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
            # crop = img_blobs[:, y1:y2+1, x1:x2+1, :] - MEAN_PIXEL
            # crop = img_blobs[:,:,:,:] - MEAN_PIXEL
            # crop_v = Variable(torch.from_numpy(crop).permute(0, 3, 1, 2).cuda(), requires_grad=False)
            # pred = unet(crop_v)
            # pred_np = pred.cpu().data.numpy()[0,0,:,:]
            # pred_np[pred_np < 0.5] = 0
            # pred_np[pred_np >= 0.5] = 1
            # pred_np = pred_np * 255
            # res = pred_np.astype(int)
            # cv2.imwrite('test.png', res)

            if y2 - y1 <= 1 or x2 - x1 <= 1:
                ipdb.set_trace()
            cv2.imwrite(os.path.join('demo', 'crop_{}_{}.png'.format(i, j)), img[y1:y2+1,x1:x2+1,:])

            rmin = y1
            rmax = y2
            cmin = x1
            cmax = x2

            im2show = np.copy(img)
            cv2.rectangle(im2show, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 2)
            cv2.imwrite(os.path.join('demo', '{}_{}.jpg'.format(i, j)), im2show)
            temp = [i, j, x1, y1, x2, y2]
            crop_set.append(temp)

    # Save
    crop_set = np.array(crop_set)
    np.save('crop', crop_set)

if __name__ == '__main__':
    test()

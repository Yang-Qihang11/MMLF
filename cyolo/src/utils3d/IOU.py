#todo IOU计算和idx赋值
import numpy
import torch
import numba

def IOU(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30

    max_num = 900000
    ind=0

    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    # todo 在else if 之前做判断index， k==K-2 and n == N-1，在之后再筛选一遍， 用【x，0】！=-10去除最后有IOU的3d框，其余用3d fea
                    # todo 即然是全匹配，直接在最后按3d框个数 把index 和 iou分为两组， 用前一组加上后一组iou为-10判断为全空
                    # if k==K-2 and n == N-1 :
                    #     iou_test_tensor = torch.FloatTensor(overlaps)  # iou_test_tensor shape: [160000,4]
                    #     tensor_index_tensor = torch.LongTensor(tensor_index)
                    #     iou_test_tensor = iou_test_tensor.permute(1, 0)
                    #     iou_test_tensor = iou_test_tensor.reshape(1, 4, 1, 9000000)
                    #     tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)
                    #     non_empty_iou_test_tensor = iou_test_tensor[:, :, :, :ind]
                    #     non_empty_tensor_index_tensor = tensor_index_tensor[:ind, :]
                    #     d3_index = non_empty_tensor_index_tensor[:, 1:2]
                    #     d3_index = d3_index.tolist()
                    #     range_0_to_22743 = list(range(22743))
                    #     int_list = [int(item) for sublist in d3_index for item in sublist]
                    #     int_list = list(set(int_list))
                    #     int_list.sort()
                    #     no_match_3d_index = [x for x in range_0_to_22743 if x not in int_list]

                    ind = ind+1



                #此处为空 iw>0 ih<0
                elif k==K-1 :
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    # no_match_3d_index_final = torch.cat((no_match_3d_index_final,torch.tensor([[k,n]])),dim=0)
                    ind = ind+1

            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n,0]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n
                # no_match_3d_index_final = torch.cat((no_match_3d_index_final, torch.tensor([[k, n]])), dim=0)
                ind = ind+1

    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind, N, K
# @numba.jit(nopython=True,parallel=True)
def IOU2(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30
    max_num = 900000
    #2d框为0

    ind=0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    # todo 在else if 之前做判断index， k==K-2 and n == N-1，在之后再筛选一遍， 用【x，0】！=-10去除最后有IOU的3d框，其余用3d fea
                    # todo 即然是全匹配，直接在最后按3d框个数 把index 和 iou分为两组， 用前一组加上后一组iou为-10判断为全空



                    ind = ind+1



                elif k==K-1 :
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    # no_match_3d_index_final = torch.cat((no_match_3d_index_final,torch.tensor([[k,n]])),dim=0)
                    ind = ind+1

            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n,0]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n
                # no_match_3d_index_final = torch.cat((no_match_3d_index_final, torch.tensor([[k, n]])), dim=0)
                ind = ind+1

    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind, N, K


#todo 忽略掉2d 3d框差异过大的融合 给出mask index
@numba.jit(nopython=True,parallel=True)
def IOU3(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index,overlaps_ratio):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30
    max_num = 900000
    #2d框为0
    ratio_overlap = overlaps_ratio
    ind=0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))

        for n in range(N):


            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    # todo 计算3d框的面积 （只在iou非0时讨论面积差异问题）若比值超过4，则认为还是不匹配
                    d3_box_area = ((boxes[n, 2] - boxes[n, 0]) *
                                   (boxes[n, 3] - boxes[n, 1]))
                    # todo 计算3d框和2d框的比值
                    ratio = d3_box_area / qbox_area
                    if ratio > 4 or ratio < 0.25:
                        overlaps[ind, 0] = -10
                        overlaps[ind, 1] = scores_3d[n, 0]
                        overlaps[ind, 2] = -10
                        overlaps[ind, 3] = dis_to_lidar_3d[n, 0]




                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    # todo 在else if 之前做判断index， k==K-2 and n == N-1，在之后再筛选一遍， 用【x，0】！=-10去除最后有IOU的3d框，其余用3d fea
                    # todo 即然是全匹配，直接在最后按3d框个数 把index 和 iou分为两组， 用前一组加上后一组iou为-10判断为全空



                    ind = ind+1



                #此处为空 iw>0 ih<0
                elif k==K-1 :
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n

                    # d3_box_area = ((boxes[n, 2] - boxes[n, 0]) *
                    #                (boxes[n, 3] - boxes[n, 1]))
                    # # todo 计算3d框和2d框的比值
                    # ratio = d3_box_area / qbox_area
                    # if ratio > 4 or ratio < 0.25:
                    #     ratio_overlap[ind, 0] = 1
                    # no_match_3d_index_final = torch.cat((no_match_3d_index_final,torch.tensor([[k,n]])),dim=0)
                    ind = ind+1

            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n,0]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n

                # d3_box_area = ((boxes[n, 2] - boxes[n, 0]) *
                #                (boxes[n, 3] - boxes[n, 1]))
                # # todo 计算3d框和2d框的比值
                # ratio = d3_box_area / qbox_area
                # if ratio > 4 or ratio < 0.25:
                #     ratio_overlap[ind, 0] = 1

                # no_match_3d_index_final = torch.cat((no_match_3d_index_final, torch.tensor([[k, n]])), dim=0)
                ind = ind+1

    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind, N, K,ratio_overlap
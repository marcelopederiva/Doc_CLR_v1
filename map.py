import numpy as np
import config_model as cfg
from utils3 import iou3d
iou_treshould = 0.7
dist_threshoulds = [0.5,1,2,4]

def mAPNuscenes(corners_predict, corners_real):
    correct_d = [0,0,0,0]
    correct_dtc = [0,0,0,0]
    i_d = [[],[],[],[]]
    iou_d = [[],[],[],[]]
    # i_05 = []
    # i_1 = []
    # i_2 = []
    # i_4 = []

    mave_x = []
    mave_y = []

    number_dtc = 0

    for d in corners_real:
        if d[-1] == 0:
            number_dtc += 1
    FP = 0

    while (len(corners_predict) > 0):
        c = True
        flag=False
        for r in range(len(corners_real)):
            
            d_x = corners_predict[0][1] - corners_real[r][0]
            d_z = corners_predict[0][3] - corners_real[r][2]

            dist = np.sqrt((d_x)*(d_x) + (d_z)*(d_z))
            iou_score = iou3d(corners_predict[0][1:-1], corners_real[r][:-1])
            # print(dist)
            for d in range(len(dist_threshoulds)):                
                if dist <= dist_threshoulds[3]:
                    # print('PREDICT: ',corners_predict[0][8],corners_predict[0][9])
                    # print('REAL: ',corners_real[r][7],corners_real[r][8],'\n')
                    mave_x.append(abs(corners_predict[0][8] - corners_real[r][7]))
                    mave_y.append(abs(corners_predict[0][9] - corners_real[r][8]))
                    # print(mave_x)
                    c = False
                    #print('Entrou 4:', dist)
                    correct_d[3] +=1
                    i_d[3].append(dist)
                    iou_d[3].append(iou_score)
                    correct_dtc[3] += 1

                    if dist <= dist_threshoulds[2]:
                        #print('Entrou 2:', dist)
                        correct_d[2] +=1
                        i_d[2].append(dist)
                        iou_d[2].append(iou_score)
                        correct_dtc[2] += 1

                    if dist <= dist_threshoulds[1]:
                        #print('Entrou 1:', dist)
                        correct_d[1] +=1
                        i_d[1].append(dist)
                        iou_d[1].append(iou_score)
                        correct_dtc[1] += 1

                    if dist <= dist_threshoulds[0]:
                        #print('Entrou 0.5:', dist)
                        correct_d[0] +=1
                        i_d[0].append(dist)
                        iou_d[0].append(iou_score)
                        correct_dtc[0] += 1
                    corners_real.pop(r)
                    # correct_d[d] += 1
                    flag = True
                    break
            if flag:
                break
        if c:
            FP += 1

        corners_predict.pop(0)
    FN = len(corners_real)

    if mave_x == []:
        mave_x=0
    if mave_y == []:
        mave_y=0


    i_d = [0 if not sublist else sublist for sublist in i_d]
    iou_d = [0 if not sublist else sublist for sublist in iou_d]
    mAP = [[],[],[],[]]
    detections = [[],[],[],[]]
    if number_dtc == 0:
        mAP = ['nd','nd','nd','nd']
        detections = ['nd','nd','nd','nd']
    else:
        for j in range(len(dist_threshoulds)):
            mAP[j] = correct_d[j]/ number_dtc
            detections[j] = correct_dtc[j] / number_dtc

    mean_i = [np.mean(dist_t) for dist_t in i_d]
    mean_iou = [np.mean(dist_t) for dist_t in iou_d]

    max_i = [np.max(dist_t) for dist_t in i_d]
    max_iou = [np.max(dist_t) for dist_t in iou_d] 
    
    TP = np.array(correct_d)
    TP = TP.astype(float)
    # print(TP)
    # print(FP)
    # print(TP + (FP + FN)/2)
    # exit()
    
    div1 = (TP + (FP + FN) / 2)
    div1[div1<= 0 ] = 10^-3
    F1 = TP / div1
    div2 = TP+ FN
    div2[div2 <= 0] = 10^-3
    Recall = TP / div2

    # mave_x = np.mean(mave_x)
 
    # mave_y = np.mean(mave_y)
    # print(mave_x)
    mave = [mave_x,mave_y]

    return mAP, mean_i, mean_iou, max_i, max_iou, detections, F1, Recall, mave


def mAPKITTI(corners_predict, corners_real):
    correct_ez = 0
    correct_med = 0
    correct_hard = 0

    corr_dtc_ez = 0
    corr_dtc_med = 0
    corr_dtc_hard = 0

    i_ez = []
    i_med = []
    i_hard = []

    number_dtc_ez = 0
    number_dtc_med = 0
    number_dtc_hard = 0
    for d in corners_real:
        if d[-1] == 0:
            number_dtc_ez += 1
        elif d[-1] == 1:
            number_dtc_med += 1
        elif d[-1] == 2:
            number_dtc_hard += 1
        # elif d[-1]==3:
        #     if (0.0<d[-2]<0.15):
        #         number_dtc_ez += 1
        #     if (0.15<d[-2]<0.30):
        #         number_dtc_med += 1
        #     if (0.30<d[-2]<0.50):
        #         number_dtc_hard += 1

    while (len(corners_predict) > 0):

        for r in range(len(corners_real)):
            # print(corners_predict[0])
            # input()
            iou_score = iou3d(corners_predict[0][:-2], corners_real[r][:-2])
            # print(iou_score)
            if (iou_score > 0.1) and (corners_predict[0][-2] == corners_real[r][-2]):
                # print(iou_score)
                if corners_real[r][-1] == 0:
                    i_ez.append(iou_score)
                    corr_dtc_ez += 1
                elif corners_real[r][-1] == 1:
                    i_med.append(iou_score)
                    corr_dtc_med += 1
                elif corners_real[r][-1] == 2:
                    i_hard.append(iou_score)
                    corr_dtc_hard += 1

            if (iou_score > iou_treshould) and (corners_predict[0][-2] == corners_real[r][-2]) :
                if corners_real[r][-1] == 0:
                    corners_real.pop(r)
                    correct_ez += 1
                elif corners_real[r][-1] == 1:
                    corners_real.pop(r)
                    correct_med += 1
                elif corners_real[r][-1] == 2:
                    corners_real.pop(r)
                    correct_hard += 1

                break
        corners_predict.pop(0)

    non_detected = len(corners_real)

    if i_ez == []:
        i_ez = 0
    if i_med == []:
        i_med = 0
    if i_hard == []:
        i_hard = 0

    if number_dtc_ez == 0:
        mAP_ez = 'nd'
        detections_ez = 'nd'
    elif number_dtc_ez != 0:
        mAP_ez = correct_ez / number_dtc_ez
        detections_ez = corr_dtc_ez / number_dtc_ez

    if number_dtc_med == 0:
        mAP_med = 'nd'
        detections_med = 'nd'
    elif number_dtc_med != 0:
        mAP_med = correct_med / number_dtc_med
        detections_med = corr_dtc_med / number_dtc_med

    if number_dtc_hard == 0:
        mAP_hard = 'nd'
        detections_hard = 'nd'
    elif number_dtc_hard != 0:
        mAP_hard = correct_hard / number_dtc_hard
        detections_hard = corr_dtc_hard / number_dtc_hard

    mean_i_ez = np.mean(i_ez)
    mean_i_med = np.mean(i_med)
    mean_i_hard = np.mean(i_hard)

    max_i_ez = np.max(i_ez)
    max_i_med = np.max(i_med)
    max_i_hard = np.max(i_hard)

    return mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard


if __name__ == '__main__':
    # occ,bb_x,bb_y,bb_z,bb_w,bb_h,bb_l,bb_rot,bb_velox,bb_veloz,bb_cls
    pred = [[1,6.229, 1.582, 20.621, 1.574, 1.417, 3.477, 1.053,-1,-2.2, 0]]
    true = [[6.19, 1.56, 20.48, 1.56, 1.42, 3.48, -2.08,-1.5,-2.0, 0]]
    
    mAP, mean_i, mean_iou, max_i, max_iou, detections, F1, Recall, mave = mAPNuscenes(pred,true)

    print(mAP,'\n',
          mean_i, '\n',
          mean_iou,'\n',
           max_i,'\n',
           max_iou, '\n',
          detections, '\n',
          F1, '\n',
          Recall, '\n',
          mave)
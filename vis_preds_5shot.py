import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image


frame = 'releasedModel'
shot = 'fiveShot'

#main_dir = os.path.join('/p300/AdaptiveMaskedProxies/Output/', frame, 'dilated_fcn_fold0/')

main_dir = os.path.join('/p300/AdaptiveMaskedProxies/Output/', frame, shot, 'dilated_fcn_fold0/')
#
# plt.figure(1); plt.figure(2); # Img overlay Gt, Pred
# plt.figure(3);plt.figure(4); # Heatmaps
# plt.figure(5); # Sprt Img overlay GT
# plt.ion()
# plt.show()

def make_dir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def vis_ada(overlay_qry_gt, overlay_qry_pred, hmaps_bg, hmaps_fg, overlay_sprt, img, file):

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)

    plt.subplot(2, 4, 1)
    plt.title('sprt0')
    plt.imshow(overlay_sprt[0])
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title('sprt1')
    plt.imshow(overlay_sprt[1])
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title('sprt2')
    plt.imshow(overlay_sprt[2])
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.title('sprt3')
    plt.imshow(overlay_sprt[3])
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title('sprt4')
    plt.imshow(overlay_sprt[4])
    plt.axis('off')


    plt.subplot(2, 4, 6)
    plt.title('qry_gt')
    plt.imshow(overlay_qry_gt)
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.title('qry_pred')
    plt.imshow(overlay_qry_pred)
    plt.axis('off')

    # plt.subplot(2, 3, 4)
    # plt.title('hmaps_bg')
    # plt.imshow(hmaps_bg)
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 5)
    # plt.title('hmaps_fg')
    # plt.imshow(hmaps_fg)
    # plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.title('qry_img')
    plt.imshow(img)
    plt.axis('off')


    saveRoot = os.path.join(main_dir, 'vis')
    make_dir(saveRoot)

    plt.savefig(os.path.join(saveRoot, file), dpi = 200)

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

def create_overlay(img, mask, colors):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    if len(colors)==3:
        mask_color[mask==colors[1],0]=255
        mask_color[mask==colors[1],1]=255
        mask_color[mask==colors[2],0]=255
    else:
        mask_color[mask==colors[1],2]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==colors[1],:]= blended_arr[mask==colors[1],:]
    return img2

for f in sorted(os.listdir(main_dir+'gt/')):
    print(f)
    sprt_img_list = []
    sprt_gt_list = []
    overlay_sprt_list = []
    gt = cv2.imread(main_dir+'gt/'+f, 0)
    pred = cv2.imread(main_dir+'pred/'+f, 0)
    img = cv2.imread(main_dir+'qry_images/'+f)[:, :, ::-1]
    hmaps_bg = cv2.imread(main_dir+'hmaps_bg/'+f, 0)
    hmaps_fg = cv2.imread(main_dir+'hmaps_fg/'+f, 0)
    for shot_idx in range(5):
        sprt_img_list.append(cv2.imread(main_dir+'sprt_images/'+f.split('.')[0]+'_shot' + str(shot_idx) + '.png')[:, :, ::-1])
        sprt_gt_list.append(cv2.imread(main_dir+'sprt_gt/'+f.split('.')[0]+'_shot' + str(shot_idx) + '.png', 0))

    gt[gt!=1]=0
    gt[gt==1]=255
    overlay_qry_gt= create_overlay(img, gt, [0,255])

    pred[pred!=1]=0
    pred[pred==1]=255
    overlay_qry_pred= create_overlay(img, pred, [0,255])

    for i in range(len(sprt_gt_list)):
        sprt_gt = sprt_gt_list[i]
        sprt_img = sprt_img_list[i]

        sprt_gt[sprt_gt!=16]=0
        sprt_gt[sprt_gt==16]=255
        overlay_sprt_list.append(create_overlay(sprt_img, sprt_gt, [0,255]))

        # plt.figure(1); plt.imshow(overlay_qry_gt); plt.figure(2); plt.imshow(overlay_qry_pred);
        # plt.figure(3); plt.imshow(hmaps_bg); plt.figure(4); plt.imshow(hmaps_fg)
        # plt.figure(5); plt.imshow(overlay_sprt);

        #vis_list = [overlay_qry_gt, overlay_qry_pred, hmaps_bg, hmaps_fg, overlay_sprt]
    vis_ada(overlay_qry_gt, overlay_qry_pred, hmaps_bg, hmaps_fg, overlay_sprt_list, img, f)




    #plt.draw()
    #plt.waitforbuttonpress(0)


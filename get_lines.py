import os


def get_video_id(path):
    video_id_dict = {}

    with open(path+'list_video_id.txt','r') as f:
        for line in f:
            line = line.rstrip()
            video = line.split(' ')
            video_id_dict[int(video[0])] = video[1]

    return video_id_dict



def get_rois(cam_id,data_path):
    cam_path = os.path.join('ROIs', 'cam_{}.txt'.format(cam_id))
    with open(cam_path, 'r') as f:
        rois=[]
        for line in f:
            line = line.rstrip()
            p0 = (int(line.split(',')[0]),int(line.split(',')[1]))
            pre=p0
            break
        for line in f:
            line = line.rstrip()
            now = (int(line.split(',')[0]),int(line.split(',')[1]))
            rois.append([pre,now])
            pre=now
        rois.append([pre,p0])

    return len(rois), rois

def get_lines(cam_id):

    directions = []  # valid direction for each line
    mov_rois = []  # corresponding RoI line for each movement

    if cam_id == 1:

        line_4 = [(543, 543), (595, 452)]
        line_3 = [(481, 387), (657, 361)]
        line_2 = [(328, 297), (409, 354)]
        line_1 = [(289, 224), (335, 275)]
        lines=[line_1,line_2,line_3,line_4]
        mov_rois = [4, 1, 1, 2]


    elif cam_id == 2:

        line_1=[(245, 274), (404, 317)]
        line_2=[(547, 267), (724, 310)]
        line_3=[(155, 128), (201, 145)]
        line_4=[(1068, 222), (1236, 214)]
        lines = [line_1, line_2, line_3, line_4]
        mov_rois = [4, 1, 4, 1]



    elif cam_id == 3:

        line_4=[(566, 394), (769, 390)]
        line_3=[(556, 508), (593, 452)]
        line_2=[(573, 319), (614, 374)]
        line_1=[(460, 299), (481, 337)]
        lines = [line_1, line_2, line_3, line_4]
        mov_rois = [3, 1, 1, 1]


    elif cam_id == 4:

        line_12=[(255, 302), (299, 351)]
        line_11=[(302, 352), (497, 523)]
        line_10=[(541, 532), (583, 814)]

        line_9=[(332, 263), (360, 207)]
        line_8=[(1103, 495),(1163, 406)]
        line_7=[(1015, 326), (1083, 310)]

        line_6=[(633, 146), (725, 173)]
        line_5=[(530, 122), (627, 146)]
        line_4=[(450, 105), (435, 131)]

        line_3=[(287, 158), (333, 139)]
        line_2=[(195, 197), (283, 159)]
        line_1=[(104, 202), (168, 201)]

        mov_rois = [4, 2, 1, 0, 4, 2, 1, 0, 4, 2, 1, 0]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]


    elif cam_id == 5:

        line_1 = [(30,  265), (141, 273)]
        line_2 = [(141, 273), (185, 250)]
        line_3 = [(185, 250), (342, 179)]

        line_4 = [(342, 179), (384, 110)]
        line_5 = [(470, 150), (550, 180)]
        line_6 = [(550, 180), (614, 209)]

        line_7 = [(1000, 320), (935, 399)]
        line_8 = [(935, 399), (894, 450)]
        line_9 = [(894, 450), (836, 516)]

        line_10 = [(397, 676), (401, 942)]
        line_11 = [(248, 409), (356, 543)]
        line_12 = [(126, 394), (248, 409)]

        directions = [3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 4]

        mov_rois = [5, 3, 2, 0, 5, 3, 2, 0, 5, 3, 2, 0]

        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]

    elif cam_id == 6:


        line_1=[(61, 289), (131, 325)]
        line_2=[(251, 266), (349, 236)]
        line_3=[(355, 231), (410, 200)]

        line_4=[(619, 177), (602, 223)]
        line_5=[(676, 207), (723, 272)]
        line_6=[(742, 281), (838, 322)]

        line_7=[(1074, 479), (1201, 474)]
        line_8=[(940, 538), (1059, 469)]
        line_9=[(786, 666),(928, 544)]

        line_10=[(318, 581), (229, 736)]
        line_11=[(186, 586), (167, 431)]
        line_12=[(172, 420), (162, 341)]

        directions = [3, 3, 3, 2, 2, 2, 4, 4, 4, 1, 1, 1]

        mov_rois = [5, 4, 2, 0, 5, 4, 2, 0, 5, 4, 2, 0]

        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]

    elif cam_id == 7:

        line_1=[(349, 449), (416, 467)]
        line_2=[(412, 469), (567, 491)]
        line_3=[(581, 493), (693, 490)]

        line_4=[(1072, 558), (1092, 511)]
        line_5=[(558, 525),(579, 493)]
        line_6=[(571, 578), (546, 530)]

        line_7=[(1113, 578), (1129, 615)]
        line_8=[(799, 581), (959, 596)]
        line_9=[(711, 581), (793, 586)]

        line_10=[(310, 558), (388, 531)]
        line_11=[(828, 517), (806, 565)]
        line_12=[(776, 488), (808, 522)]

        directions = [3, 3, 3, 2, 2, 2, 1, 4, 4, 1, 1, 1]

        mov_rois = [4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1]

        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]

    elif cam_id == 8:

        line_6=[(250, 353), (248, 595)]
        line_5=[(867, 688), (730, 932)]
        line_4=[(1790, 736), (1772, 935)]

        line_3=[(1793, 531), (1773, 726)]
        line_2=[(1021, 240), (1186, 209)]
        line_1=[(916, 74), (986, 161)]

        directions=[3,3,2,2,1,1]

        mov_rois = [5, 3, 1, 5, 3, 1]

        lines = [line_1, line_2, line_3, line_4, line_5, line_6]

    elif cam_id == 9:

        line_1=[(18, 423), (219, 474)]
        line_2=[(516, 453), (740, 347)]
        line_3=[(738, 351), (729, 264)]

        line_4=[(753, 174), (799, 278)]
        line_5=[(823, 289), (1026, 339)]
        line_6=[(1072, 392), (1270, 368)]

        line_7=[(1459, 381), (1700, 357)]
        line_8=[(1340, 451), (1458, 379)]
        line_9=[(1240, 610), (1334, 456)]

        line_10=[(884, 705),(983, 1019)]
        line_11=[(559, 501), (889, 666)]
        line_12=[(313, 505),(557, 494)]

        directions = [3, 1, 1, 2, 3, 3, 2, 2, 2, 4, 4, 4]

        mov_rois = [4, 2, 1, 0, 4, 2, 1, 0, 4, 2, 1, 0]

        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]

    elif cam_id == 10:

        line_1=[(28, 462), (268, 596)]
        line_2=[(272, 603), (849, 660)]
        line_3=[(873, 660), (1246, 615)]

        directions = [3, 3, 3]

        mov_rois = [3, 2, 1]

        lines =  [line_1, line_2, line_3]

    elif cam_id == 11:

        line_1=[(27, 578), (216, 631)]
        line_2=[(226, 627), (762, 621)]
        line_3=[(771, 627), (1112, 593)]

        directions = [3, 3, 3]

        mov_rois = [3, 2, 1]

        lines = [line_1, line_2, line_3]

    elif cam_id == 12:

        line_1=[(242, 337), (227, 402)]
        line_2=[(393, 386), (808, 372)]
        line_3=[(819, 372), (999, 341)]

        directions = [2, 3, 3]

        mov_rois = [4, 3, 2]

        lines = [line_1, line_2, line_3]

    elif cam_id == 13:

        line_1=[(418, 389), (600, 413)]
        line_2=[(701, 491), (976, 477)]
        line_3=[(972, 379), (1182, 369)]

        directions = [3, 3, 3]

        mov_rois = [3, 2, 1]

        lines = [line_1, line_2, line_3]

    elif cam_id == 14:

        line_1=[(242, 1018), (531, 1377)]
        line_2=[(1598, 1312), (2079, 1531)]
        mov_rois = [3, 1]
        lines = [line_1, line_2]

    elif cam_id == 15:

        line_1=[(965, 575), (1286, 433)]
        line_2=[(845, 312), (1045, 248)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    elif cam_id == 16:
        line_1=[(236, 657), (947, 665)]
        line_2=[(824, 211), (1171, 216)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    elif cam_id == 17:
        line_1=[(796, 843), (1519, 721)]
        line_2=[(812, 227), (1209, 216)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    elif cam_id == 18:
        line_1=[(225, 832), (744, 856)]
        line_2=[(923, 285), (1175, 289)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    elif cam_id == 19:
        line_1=[(585, 786), (1039, 800)]
        line_2=[(1048, 647), (1491, 644)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    elif cam_id == 20:
        line_1=[(220, 586), (855, 672)]
        line_2=[(1018, 655), (1525, 694)]
        mov_rois = [2, 0]
        lines = [line_1, line_2]

    return len(lines), lines, directions, mov_rois



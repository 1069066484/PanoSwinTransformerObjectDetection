import os
import shutil
import pickle as pkl
import cv2
import numpy as np
import copy
# hw0805/第二批10-7/哈道里二场村南(1.jpg


def run_(folder, images_folder, label_folder):
    np.random.seed(0)
    images_folder_train = os.path.join(images_folder, "train")
    images_folder_test = os.path.join(images_folder, "valid")
    label_folder_train = os.path.join(label_folder, "train")
    label_folder_test = os.path.join(label_folder, "valid")
    for f in [images_folder, label_folder, images_folder_train,
              images_folder_test, label_folder_train, label_folder_test]:
        if not os.path.exists(f):
            os.mkdir(f)
    i_cnt = 0

    np.random.seed(0)
    labels = set()
    path_cache = os.path.join(folder, "test_folders.pkl")
    if not os.path.exists(path_cache):
        label_folders = set()
        for root, dirs, files in os.walk(folder):
            for file in files:
                if not file.endswith(".txt"): continue
                file = os.path.join(root, file)
                with open(file, 'r') as f:
                    txt = f.read()
                lines = [s for s in txt.split("\n") if len(s) != 0]
                for l in lines:
                    labels.add(l.split(' ')[0])
                cls_name = os.path.split(os.path.dirname(file))[-1]
                if len(label_folders) < 15 and np.random.rand() < 0.2 and cls_name not in label_folders:
                    label_folders.add(cls_name)
        label_folders = set(label_folders)
        with open(path_cache, 'wb') as f:
            pkl.dump(label_folders, f)
    else:
        with open(path_cache, 'rb') as f:
            label_folders = pkl.load(f)
    print("Test label folders: {}".format(label_folders))
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.endswith(".txt"): continue
            i_cnt += 1
            is_train = (i_cnt % 10 != 0)
            file_ori = file
            file = os.path.join(root, file)
            cls_name = os.path.split(os.path.dirname(file))[-1]
            is_train = cls_name not in label_folders
            txt_file = os.path.splitext(file)[0] + '.txt'
            jpg_file = os.path.splitext(file.replace("_mask", ""))[0] + '.jpg'
            pref = os.path.split(root)[-1] + "__"
            if not is_train:
                target_jpg_file = os.path.join(images_folder_test, os.path.split(jpg_file)[-1])
                if not os.path.exists(target_jpg_file):
                    shutil.copy(jpg_file, os.path.join( images_folder_test,
                                                        pref + os.path.split(jpg_file)[-1]))
                shutil.copy(txt_file, os.path.join(label_folder_test,
                                                   pref + file_ori.replace("_mask", "")))
                print("copy {} to valid folder".format([jpg_file, txt_file]))
            else:
                target_jpg_file = os.path.join(images_folder_train, os.path.split(jpg_file)[-1])
                if not os.path.exists(target_jpg_file):
                    shutil.copy(jpg_file, os.path.join(images_folder_train,
                                                       pref + os.path.split(jpg_file)[-1]))
                shutil.copy(txt_file, os.path.join(label_folder_train,
                                                   pref +  file_ori.replace("_mask", "")))
                print("copy {} to train folder".format([jpg_file, txt_file]))


# run("hw0805", "data_trainable/images", "data_trainable/labels")

def run(folder, images_folder, label_folder, all_rd=False):
    np.random.seed(0)
    images_folder_train = os.path.join(images_folder, "train")
    images_folder_test = os.path.join(images_folder, "valid")
    label_folder_train = os.path.join(label_folder, "train")
    label_folder_test = os.path.join(label_folder, "valid")
    for f in [images_folder, label_folder, images_folder_train,
              images_folder_test, label_folder_train, label_folder_test]:
        if not os.path.exists(f):
            os.mkdir(f)
    i_cnt = 0
    is_train = True
    np.random.seed(0)
    labels = dict()
    labels_sum = dict()
    path_cache = os.path.join(folder, "test_folders.pkl")

    label_folders =  {'10a3q6a3q2a3q3a3q71',
                        '23010200000011_kbcl001_mnopqqrs_1631083276750',
                        '230103500000000028_kbcl_mnopqqrs_1631083442233',
                        '230103908000000677_kbcl_mnopqqrs_1631084844343',
                        '230104908000000114_kbcl001_mnopqqrs_1630542279727',
                        '23010900000001_tuXea3q10a3q10hvpca3q10a3q10GtfJ1457zmypgjkbcl001_mnopqqrs_1628578060406',
                        '230109908000000038_kbcl_mnopqqrs_1631152536383',
                        '230110908000000115_a3q0a3q1001_a3q2a3q3a3q4a3q5_1625226099342',
                        '230111908000000395',
                        '230111908000000481_kbcl001_mnopqqrs_1630284268298',
                        '230111908001900235_kbcl_mnopqqrs_1630284377903',
                        '230112500000000112_kbcl_mnopqqrs_1630629695808',
                        '230112908000000149_kbcl_mnopqqrs_1630569014975',
                        '23012300000087_kbcl_mnopqqrs_1630289363606',
                        '230123600000000342_kbcl001_mnopqqrs_1630568823637',
                        '230123908000000370qFBBC_FBBCiDwIhH_mn_gHVm_gHVmXa3q24cfXu',
                        '230124908000000060_tucfqvwxgy_kbcl_mnopqqrs_1629073862626',
                        '230126908000000111_kbcl_mnopqqrs_1630459415471',
                        '230127908000000140_kbcl_mnopqqrs_1628219272486',
                        '230128908000000011_kbcl_mnopqqrs_1630460647489',
                        '230129908000000060',
                        '230182700000201321_tuXniYTDBEGHkbcl_mnopqqrs_1629161197245',
                        '230182908000000043_kbcl_mnopqqrs_1628847094806',
                        '230182908000000227',
                        '230183908000000349',
    }
    if len(label_folders) == 0:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if not file.endswith(".txt"): continue
                file = os.path.join(root, file)
                with open(file, 'r') as f:
                    txt = f.read()
                lines = [s for s in txt.split("\n") if len(s) != 0]
                cls_name = os.path.split(os.path.dirname(file))[-1]
                if cls_name not in labels:
                    labels[cls_name] = dict()
                for l in lines:
                    label = l.split(' ')[0]
                    if label not in labels[cls_name]:
                        labels[cls_name][label] = 0
                    labels[cls_name][label] += 1
                    if label not in labels_sum:
                        labels_sum[label] = 0
                    labels_sum[label] += 1
        print("labels: {}".format(labels))
        print("labels_sum: {}".format(labels_sum))
        labels_sum_ = copy.deepcopy(labels_sum)
        t = 0
        while True:
            t += 1
            for k in labels_sum_: labels_sum_[k] = 0
            label_folders = set(np.random.choice(list(labels.keys()), 15, False))
            for cls_name in label_folders:
                for l in labels[cls_name]:
                    labels_sum_[l] += labels[cls_name][l]
            print("time: {}, Folders: {}".format(t, label_folders))
            ok = True
            for k in labels_sum:
                print("key: {}, all: {}, curr: {}".format(k, labels_sum[k], labels_sum_[k]))
                if labels_sum[k] >= 100 and labels_sum_[k] <= 1:
                    ok = False
            if not ok:
                print("Bad split, continue")
                continue
            cmd = input("Input a to confirm")
            if cmd == 'a':
                print("Folders: {} confirmed, saved in path_cache".format(label_folders))
                break

        with open(path_cache, 'wb') as f:
            pkl.dump(label_folders, f)


    print("Test label folders: {}".format(label_folders))
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.endswith(".txt"): continue
            i_cnt += 1
            is_train = (i_cnt % 20 != 0)
            file_ori = file
            file = os.path.join(root, file)
            cls_name = os.path.split(os.path.dirname(file))[-1]
            if not all_rd: is_train = cls_name not in label_folders
            txt_file = os.path.splitext(file)[0] + '.txt'
            jpg_file = os.path.splitext(file.replace("_mask", ""))[0] + '.jpg'
            pref = os.path.split(root)[-1] + "____"
            if not is_train:
                target_jpg_file = os.path.join(images_folder_test, os.path.split(jpg_file)[-1])
                if not os.path.exists(target_jpg_file):
                    shutil.copy(jpg_file, os.path.join( images_folder_test,
                                                        pref + os.path.split(jpg_file)[-1]))
                shutil.copy(txt_file, os.path.join(label_folder_test,
                                                   pref + file_ori.replace("_mask", "")))
                print("copy {} to valid folder".format([jpg_file, txt_file]))
            else:
                target_jpg_file = os.path.join(images_folder_train, os.path.split(jpg_file)[-1])
                if not os.path.exists(target_jpg_file):
                    shutil.copy(jpg_file, os.path.join(images_folder_train,
                                                       pref + os.path.split(jpg_file)[-1]))
                shutil.copy(txt_file, os.path.join(label_folder_train,
                                                   pref +  file_ori.replace("_mask", "")))
                print("copy {} to train folder".format([jpg_file, txt_file]))



d = {'ACdist':'交流配电箱',
     '交流配电箱':'ACdist',
     'batteries':'蓄电池组',
     '蓄电池组':'batteries',
     'wirelesscabinet':'无线机柜',
     '无线机柜':'wirelesscabinet',
     'heat-tubeac':'热管空调',
     '热管空调':'heat-tubeac',
     'ladderbatteries':'梯式蓄电池组',
     '梯式蓄电池组':'ladderbatteries',
     'generalcabinet':'综合柜',
     '综合柜':'generalcabinet',
     'powercabinet':'开关电源柜',
     '开关电源柜':'powercabinet',
     'FSU':'FSU','groundwire':'接地排',
     '接地排':'groundwire',
     'ac':'空调',
     '空调':'ac',
     'othercabinet':'其他机柜',
     '其他机柜':'othercabinet',
     'fansys':'新风系统',
     '新风系>统':'fansys',
     '新风系统':'fansys',
     'hangingac':'挂式空调',
     '挂式空调':'hangingac',
     'groundbox':'接地箱',
     '接地箱':'groundbox',
     'transformer':'变压器',
     '变压器':'transformer',
     'powerbox':'室外市电引入电源箱',
     '室外市电引入电源箱':'powerbox',
     'Libatteries':'锂电池组',
     '锂电池组':'Libatteries',
     'unifiedcabinet':'室外一体化柜',
     '室外一体化柜':'unifiedcabinet',
     'monitorbox':'动环监控箱',
     '动环监控箱':'monitorbox',
     'acexternal':'空调外机',
     '空调外机':'acexternal',
     'cabinet':'cabinet',
     'DCdistribution':'直流>配电箱',
     '直流>配电箱':'DCdistribution',
     '直流配电箱':'DCdistribution',
     'ladderbattery':'梯式蓄电池',
     '梯式蓄电池':'ladderbattery',
     '电池柜': 'batterycabinet',
    'batterycabinet':'电池柜'
     }


def txt_label_trans(folder):
    np.random.seed(0)
    l = ['ACdist', 'batteries', 'wirelesscabinet', 'heat-tubeac', 'ladderbatteries', 'generalcabinet', 'powercabinet', 'FSU', 'groundwire', 'ac', 'othercabinet', 'fansys', 'hangingac', 'groundbox', 'transformer', 'powerbox', 'Libatteries', 'unifiedcabinet', 'monitorbox', 'acexternal', 'cabinet', 'DCdistribution', 'ladderbattery', 'batterycabinet']
    ld = dict([(k,str(l.index(k))) for k in l])
    i = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            content = ""
            file = os.path.join(root, file)
            if not file.endswith(".txt"): continue
            print("processing {}".format(file))
            with open(file, 'r', encoding='utf8') as f:
                txt = f.read()
            lines = [s for s in txt.split("\n") if len(s) != 0]
            for l in lines:
                l = l.split(' ')
                #if l[0] not in d:
                #    d[l[0]] = len(d)
                content += " ".join([ld[str(d[l[0]])]] + l[1:]) + '\n'
            print("{}: {} ok".format(i, file))
            i += 1
            with open(file, 'w', encoding='utf8') as f:
                f.write(content)
    with open(os.path.join(folder, "d.out"), 'w') as f:
        f.write("{}\n{}\n{}\n".format(d, l, ld))
    print(d)

# txt_label_trans(r"data_trainable/labels")


"""

{'交流配电箱': 0, '蓄电池组': 1, '无线机柜': 2, '热管空调': 3, '梯式蓄电池组': 4, '综合柜': 5, '开关电源柜': 6, 'FSU': 7, '接地排': 8, '空调': 9, '其他机柜': 10, '新风系>统': 11, '挂式空调': 12, '接地箱': 13, '变压器': 14, '室外市电引入电源箱': 15, '锂电池组': 16, '室外一体化柜': 17, '动环监控箱': 18, '空调外机': 19, 'cabinet': 20, '直流>配电箱': 21, '梯式蓄电池': 22}
['交流配电箱', '蓄电池组', '无线机柜', '热管空调', '梯式蓄电池组', '综合柜', '开关电源柜', 'FSU', '接地排', '空调', '其他机柜', '新风系>统', '挂式空调', '接地箱', '变压器', '室外市电引入电源箱', '锂电池组', '室外一体化柜', '动环监控箱', '空调外机', 'cabinet', '直流>配电箱', '梯式蓄电池']

"""
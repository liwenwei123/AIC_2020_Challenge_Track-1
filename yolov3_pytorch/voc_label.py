import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', 'val']

classes = ["car", "truck"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(in_xml_file, out_xml_dir):

    in_file = open(in_xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for img in root.iter('image'):
        name = img.get('name')

        out_file = open(out_xml_dir + '%s.txt' % (name.split('.')[0]), 'a')
        w = int(img.get('width'))
        h = int(img.get('height'))
        box_lst = img.iter('box')
        for bo in box_lst:
            cls = bo.get('label')
            cls_id = classes.index(cls)
            b = (float(bo.get('xtl')), float(bo.get('xbr')),
                 float(bo.get('ytl')), float(bo.get('ybr')))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')




origin_annotations_path = 'data/Annotations_origin/'  # original xml annotations
dest_annotations_path = 'data/labels/'  # output txt annotations
if not os.path.exists('data/labels/'):
    os.makedirs('data/labels/')
origin_xml_path_lst = [os.path.join(
    origin_annotations_path, x) for x in os.listdir(origin_annotations_path)]
for origin_xml_path in origin_xml_path_lst:
    convert_annotation(origin_xml_path, dest_annotations_path)


for image_set in sets:
    image_ids = open('data/ImageSets/%s.txt' %
                     (image_set)).read().strip().split()
    list_file = open('data/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data/images/%s.jpg\n' % (image_id))
    list_file.close()

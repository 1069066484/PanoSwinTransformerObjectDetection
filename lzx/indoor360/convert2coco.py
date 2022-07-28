import json


split = 'train'
input_file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\{}.json".format(split)
target_file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\coco_format\{}.json".format(split)


with open(input_file) as f:
    in_d = json.load(f)

imid_old2new = {}
# print(list(in_d.keys())) # ['images', 'type', 'annotations', 'categories']
print(list(in_d['images'][0].keys())) # ['file_name', 'height', 'width', 'id']
print(list(in_d['annotations'][0].keys())) # ['area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id', 'ignore', 'segmentation']
for item in in_d['images']:
    imid_old2new[item['id']] = len(imid_old2new)
    item['id'] = imid_old2new[item['id']]

for item in in_d['annotations']:
    item['image_id'] = imid_old2new[item['image_id']]

with open(target_file, 'w') as f:
    json.dump(in_d, f)

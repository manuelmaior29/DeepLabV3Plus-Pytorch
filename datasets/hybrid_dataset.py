from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple

class SemanticLabelMapper():
    
    ID_TO_STRING = {
        'common': {
            0: 'road',
            1: 'sidewalk',
            2: 'building',
            3: 'wall',
            4: 'fence',
            5: 'trafficlight',
            6: 'trafficsign',
            7: 'vegetation',
            8: 'terrain',
            9: 'pedestrian',
            10: 'rider',
            11: 'car',
            12: 'truck',
            13: 'bus',
            14: 'motorcycle',
            15: 'bicycle',
            16: 'background'
        }
    }

    ID_TO_COLOR = {
        'common': {
            0: (70, 70, 70),
            1: (100, 40, 40),
            2: (55, 90, 80),
            3: (220, 20, 60),
            4: (153, 153, 153),
            5: (157, 234, 50),
            6: (128, 64, 128),
            7: (244, 35, 232),
            8: (107, 142, 35),
            9: (0, 0, 142),
            10: (102, 102, 156),
            11: (220, 220, 0),
            12: (70, 130, 180),
            13: (81, 0, 81),
            14: (150, 100, 100),
            15: (230, 150, 140),
            16: (0, 0, 0)
        }
    }

    MAPPING = {
        'carla_to_common': [
        #   0   1  2  3  4  5  6   7  8  9  10 11  12 13  14  15  16  17  18  19  20  21  22  23  24 25  26  27  28
            16, 0, 1, 2, 3, 4, 16, 5, 6, 7, 8, 16, 9, 10, 11, 12, 13, 16, 14, 15, 16, 16, 16, 16, 0, 16, 16, 16, 16
        ],
        'cityscapes_to_common': [
        #   0   1   2   3   4   5   6   7  8  9   10  11 12 13 14  15  16  17  18  19 20 21 22 23  24 25  26  27  28  29  30  31  32  33  34    
            16, 16, 16, 16, 16, 16, 16, 0, 1, 16, 16, 2, 3, 4, 16, 16, 16, 16, 16, 5, 6, 7, 8, 16, 9, 10, 11, 12, 13, 16, 16, 16, 14, 15, 16   
        ]
    }

    def __init__(self, type=None) -> None:
        super().__init__()
        self.type = type

    def __map_value(self, pixel):
        return SemanticLabelMapper.MAPPING[self.type][pixel]

    def map_image(self, input):
        return np.vectorize(self.__map_value)(input)
    
    def map_from_dir(self, src_path, dst_path, extension):
        for file in tqdm(os.listdir(src_path)):
            if file.endswith(extension):
                src_image_path = f'{src_path}/{file}'
                dst_image_path = f'{dst_path}/{file}'
                src_image = np.array(Image.open(src_image_path, 'r'))
                dst_image = self.map_image(src_image)            
                dst_image = Image.fromarray(np.uint8(dst_image), 'L')
                dst_image.save(dst_image_path)

class HybridDataset(Dataset):

    HybridDatasetClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        HybridDatasetClass('road',          0, 0, 'void', 0, False, True, (0, 0, 0)),
        HybridDatasetClass('sidewalk',      1, 1, 'void', 0, False, True, (0, 0, 230)),
        HybridDatasetClass('building',      2, 2, 'void', 0, False, True, (0, 0, 110)),
        HybridDatasetClass('wall',          3, 3, 'void', 0, False, True, (0, 80, 100)),
        HybridDatasetClass('fence',         4, 4, 'void', 0, False, True, (0, 60, 100)),
        HybridDatasetClass('trafficlight',  5, 5, 'void', 0, False, True, (111, 74, 0)),
        HybridDatasetClass('trafficsign',   6, 6, 'void', 0, False, True, (81, 0, 81)),
        HybridDatasetClass('vegetation',    7, 7, 'flat', 1, False, False, (128, 64, 128)),
        HybridDatasetClass('terrain',       8, 8, 'flat', 1, False, False, (244, 35, 232)),
        HybridDatasetClass('pedestrian',    9, 9, 'flat', 1, False, True, (250, 170, 160)),
        HybridDatasetClass('rider',         10, 10, 'flat', 1, False, True, (230, 150, 140)),
        HybridDatasetClass('car',           11, 11, 'construction', 2, False, False, (70, 70, 70)),
        HybridDatasetClass('truck',         12, 12, 'construction', 2, False, False, (102, 102, 156)),
        HybridDatasetClass('bus',           13, 13, 'construction', 2, False, False, (190, 153, 153)),
        HybridDatasetClass('motorcycle',    14, 14, 'construction', 2, False, True, (180, 165, 180)),
        HybridDatasetClass('bicycle',       15, 15, 'construction', 2, False, True, (150, 100, 100)),
        HybridDatasetClass('background',    16, 16, 'construction', 2, False, True, (150, 120, 90)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root_path, input_dir, target_dir, transform=None, type='real') -> None:
        super(HybridDataset, self).__init__()
        self.root_path = root_path
        self.input_data = input_dir
        self.target_data = target_dir
        self.transform = transform
        self.type = type
    
    def __len__(self):    
        input_file_list = os.listdir(os.path.join(self.root_path, self.input_data))
        target_file_list = os.listdir(os.path.join(self.root_path, self.target_data))
        
        input_length = len(input_file_list)
        target_length = len(target_file_list)
        
        if target_length == input_length:
            return input_length
        
        exit(1)

    def __getitem__(self, idx):
        img_path_ipt_patch = os.path.join(self.root_path, self.input_data, f"{self.type}_rgb_{idx}.png")
        img_path_tgt_patch = os.path.join(self.root_path, self.target_data, f"{self.type}_semantic_segmentation_{idx}.png")
        
        ipt_patch = Image.open(img_path_ipt_patch, 'r').convert('RGB')
        tgt_patch = Image.open(img_path_tgt_patch, 'r',)
        
        if self.transform:
            ipt_patch, tgt_patch = self.transform(ipt_patch, tgt_patch)
        return ipt_patch, tgt_patch

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

def perform_image_mapping(src_path, dst_path, mapping_type):
    slm = SemanticLabelMapper(mapping_type)
    slm.map_from_dir(src_path=src_path, dst_path=dst_path, extension='.png')
    
def visualize_class_distribution(dataset):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    classes_keys = list(SemanticLabelMapper.ID_TO_STRING['common'].keys())
    classes_labels = list(SemanticLabelMapper.ID_TO_STRING['common'].values())
    classes_distribution = {}
    for key in classes_keys:
        classes_distribution[key] = 0

    pixel_count = 0
    batch_idx = 0
 
    for _, target_batch in tqdm(dataloader):
        for target_map in target_batch:
            flattened_target_map = (torch.flatten(target_map)).long()
            labels_count = torch.bincount(flattened_target_map)
            for i in range(len(labels_count)):
                if labels_count[i] != 0:
                    classes_distribution[i] += labels_count[i].item()

        pixel_count += 1
        batch_idx += 1

    keys = list(classes_labels)
    values = list(classes_distribution.values())
    values_count = sum(values)

    # Plot the data using a bar plot
    plt.bar(keys, values)
    plt.xticks(keys, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class distribution')

    for i, v in enumerate(values):
        plt.text(i, v, str(round(v/values_count*100, 3)) + '%', ha='center')

    plt.show()

src_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\synthetic\train\semantic_segmentation'
dst_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\synthetic\train\semantic_segmentation_mapped'
mapping_type = 'carla_to_common'
# perform_image_mapping(src_path, dst_path, mapping_type)

type = 'synthetic'
dataset_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\synthetic\train'
# visualize_class_distribution(dataset_path=dataset_path, type=type)
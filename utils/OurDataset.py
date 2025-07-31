import torch
import pickle
import monai.transforms as mtransforms
import torch.nn.functional as F
import pickle

import monai.transforms as mtransforms
import torch
import torch.nn.functional as F


def mask_to_one_hot(mask):
    mask = mask.squeeze(0).type(torch.long)
    one_hot_mask = F.one_hot(mask, num_classes=57)  # all classes are 57
    one_hot_mask = one_hot_mask.permute(3, 0, 1, 2)[1:]
    return one_hot_mask.float()


# train and test csv formalation
def pair_form(input_dict, target_key):
    target_list = input_dict.get(target_key, [])
    non_target_lists = []
    for key, value in input_dict.items():
        if key != target_key:
            non_target_lists.extend(value)

    return target_list, non_target_lists


# extract the corresponding subtypes
def extract_subgroups(pairs, mode='NCAD'):
    def extract_fold(pair, label_0, label_1):
        delete_id = ['037_S_4706', '023_S_4796', '012_S_4094', '109_S_2200', '141_S_0851', '072_S_4063']
        return [item for item in pair if (item[1] == label_0 or item[1] == label_1) and (item[0] not in delete_id)]

    if mode == 'NCAD':
        return {
            0: extract_fold(pairs[0], '0', '1'),
            1: extract_fold(pairs[1], '0', '1'),
            2: extract_fold(pairs[2], '0', '1'),
            3: extract_fold(pairs[3], '0', '1'),
            4: extract_fold(pairs[4], '0', '1')
        }
    else:
        return {
            0: extract_fold(pairs[0], '3', '4'),
            1: extract_fold(pairs[1], '3', '4'),
            2: extract_fold(pairs[2], '3', '4'),
            3: extract_fold(pairs[3], '3', '4'),
            4: extract_fold(pairs[4], '3', '4')
        }




class OurDataset(torch.utils.data.Dataset):
    """
    Dataset class

    Parameters:
        root - - : (str) Path of the root folder
        mode - - : (str) {'train' or 'test'} Part of the dataset that is loaded
        seed - - : (int) random seed
    """

    def __init__(self,
                 root=1,
                 mtype='NCAD',
                 mode="train",
                 type="single",

                 fold=0,

                 ):
        # basic initialize
        self.mode = mode
        if root==1:
            self.basic_dir = '/data/chwang/AtlasAAAI/'
        else:
            self.basic_dir = '/mnt/miah203/chwang/AtlasProject/data/'

        # datapairs organization
        if type=='single':
            self.data_pairs = pickle.load(open(self.basic_dir+'all_mri_dict.pkl', 'rb'))
        elif type == 'multi_mri':
            self.data_pairs = pickle.load(open(self.basic_dir + 'pure_mri_dict.pkl', 'rb'))
        elif type == 'multi_pet':
            self.data_pairs = pickle.load(open(self.basic_dir + 'paired_dict.pkl', 'rb'))
        self.organized_pairs = extract_subgroups(self.data_pairs, mtype)
        self.test_pairs, self.train_pairs = pair_form(self.organized_pairs, fold)

        # data preprocesing
        # [MRI]
        self.basic_mri_transform = mtransforms.Compose(
            [mtransforms.LoadImage(image_only=True),
             mtransforms.EnsureChannelFirst(),
             mtransforms.SqueezeDim(),
             mtransforms.EnsureChannelFirst(),
             mtransforms.EnsureType(),
             mtransforms.ScaleIntensityRangePercentiles(lower=0, upper=99, b_min=-1.0,
                                                        b_max=1.0, clip=True, relative=False),
             mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192))
             ])
        # [PET]
        self.basic_pet_transform = mtransforms.Compose(
            [mtransforms.LoadImage(image_only=True),
             mtransforms.EnsureChannelFirst(),
             mtransforms.SqueezeDim(),
             mtransforms.EnsureChannelFirst(),
             mtransforms.EnsureType(),
             mtransforms.ScaleIntensityRangePercentiles(lower=0, upper=99, b_min=-1.0,
                                                        b_max=1.0, clip=True, relative=False),
             mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192))
             ])

        # [Atlas]
        self.basic_atlas_transform = mtransforms.Compose(
            [mtransforms.LoadImage(image_only=True),
             mtransforms.EnsureChannelFirst(),
             mtransforms.SqueezeDim(),
             mtransforms.EnsureChannelFirst(),
             mtransforms.EnsureType(),
             mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192)),
             mtransforms.Resize(spatial_size=(24, 28, 24), mode='nearest'),
             mtransforms.Flip(spatial_axis=-3)
             #                  mtransforms.Flip(spatial_axis=-2)
             ])


        # basic formulation
        self.type = type
        if mode == "train":
            ratio=1
            half_ratio=ratio/2
            self.imgs = self.train_pairs[:int(half_ratio*len(self.train_pairs))]+self.train_pairs[int((1-half_ratio)*len(self.train_pairs)):]

        elif mode == "test":
            self.imgs = self.test_pairs

    def __getitem__(self, index):
        #  subject id
        subject_id = self.imgs[index][0]

        #   labels formalation
        if self.imgs[index][1] == '0':
            label = 0
            tlabel = 'NC'
        elif self.imgs[index][1] == '1':
            label = 1
            tlabel = 'AD'
        elif self.imgs[index][1] == '3':
            label = 0
            tlabel = 'SMCI'
        elif self.imgs[index][1] == '4':
            label = 1
            tlabel = 'PMCI'

        #  mri_Data organization
        mri_data_path = self.basic_dir+'MRI/' + tlabel + '/' + self.imgs[index][0] + '.nii.gz'
        A = self.basic_mri_transform(mri_data_path)
        atlas_data_path =  self.basic_dir+'flirt_atlas/' + tlabel + '/' + self.imgs[index][0] + '.nii.gz'
        atlas = self.basic_atlas_transform(atlas_data_path)
        # pet_data organization
        if self.type == 'multi_pet':
            #   PET image
            pet_data_path =  self.basic_dir+'Final_PET/' + tlabel + '/' + self.imgs[index][0] + '.nii.gz'
            B = self.basic_pet_transform(pet_data_path)
            return {'idx_lb': subject_id,
                    'mri_x_lb': A,
                    'pet_x_lb': B,
                    'atlas_x_lb': mask_to_one_hot(atlas),
                    'y_lb': label,
                    }
        else:
            return {'idx_lb': subject_id,
                    'x_lb': A,
                    'atlas_x_lb': mask_to_one_hot(atlas),
                    'y_lb': label}

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)

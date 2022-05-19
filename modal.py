from importlib_metadata import re
from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
    NormalizeIntensityd,
    CenterSpatialCropd
)
from monai.networks.nets import UNet
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
from matplotlib.animation import FuncAnimation, ArtistAnimation
from monai.inferers import sliding_window_inference
import time
import torch
# print_config()
import argparse
import os
import nibabel as nib
slice_number = 176
examples = '/home/binbo/modal/label-sample.nii.gz'
model_dir = '/home/binbo/modal'
spatial_size = [128, 160, 176]
file_save_path = '/home/binbo/save_path'


def id(file):
    id_files = file.split('/')[4].split('.')[0]
    return id_files


def readnii(nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    data = nib.load(nifty_file)
    affine = data.affine
    header = data.header
    image = np.array(data.get_fdata())
    # print(image.shape)
    return image, affine, header


def cut(files):
    print('='*68)
    # return print(files)
    id = files.split('/')[4].split('.')[0]
    seg, seg_affine, seg_header = readnii(examples)
    img, affine, header = readnii(files)

    skip = img.shape[2] % slice_number
    # skip = 0
    batchSize = int(img.shape[2]/slice_number)
    os.makedirs(f'{file_save_path}/images/val/{id}/', exist_ok=True)
    os.makedirs(f'{file_save_path}/labels/val/{id}/', exist_ok=True)
    for i in range(batchSize):
        converted_array = np.array(
            img[:, :, i*slice_number+skip:(i+1)*slice_number+skip], dtype=np.float32)
        nifti_file = nib.Nifti1Image(converted_array, affine, header)

        seg_converted_array = np.array(
            seg[:, :, i*slice_number+skip:(i+1)*slice_number+skip], dtype=np.float32)
        seg_nifti_file = nib.Nifti1Image(
            seg_converted_array, seg_affine, seg_header)

        nib.save(
            nifti_file, f'{file_save_path}/images/val/{id}/{id}-{i}.nii.gz')

        nib.save(seg_nifti_file,
                 f'{file_save_path}/labels/val/{id}/{id}-{i}-label.nii.gz')


def transforms(files):

    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # crop CropForeground
            CropForegroundd(keys=['image', 'label'], source_key='image'),

            # augmentation
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-135,
                a_max=215,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CenterSpatialCropd(keys=["image", "label"],
                               roi_size=(320, 416, 176)),
            # pad if the image is smaller than patch
            Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=['image', 'label'])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # crop CropForeground
            CropForegroundd(keys=['image', 'label'], source_key='image'),

            # intensity
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-135,
                a_max=215,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CenterSpatialCropd(keys=["image", "label"],
                               roi_size=(320, 416, 176)),
            # pad if the image is smaller than patch
            Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=['image', 'label'])
        ]
    )

    id_files = id(files)
    data_dir = file_save_path

    path_test_volumes = sorted(
        glob(os.path.join(data_dir, "images/val", id_files, "*.nii.gz")))
    path_test_segmentation = sorted(
        glob(os.path.join(data_dir, "labels/val", id_files, "*.nii.gz")))

    test_files = [{"image": image_name, "label": label_name}
                  for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]

    val_ds = CacheDataset(
        data=test_files, transform=val_transforms, cache_rate=1.0)
    return val_ds


def detect(files):
    threshold = 50
    id_files = id(files)
    time_cut = time.time()
    print('===============  Begin   ==================')
    cut(files)
    time_cut = time.time() - time_cut
    print('===============Time Cut==================')
    print(time_cut)
    print('---------------------------------------------')
    time_modal = time.time()
    print('===============  Begin modal ==================')
    val_ds = transforms(files)
    case_num = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputResults = []

    model = UNETR(
        in_channels=1,
        out_channels=11,
        img_size=spatial_size,
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0
    ).to(device)

    torch.backends.cudnn.benchmark = True
    model.load_state_dict(torch.load(os.path.join(
        model_dir, "best_metric_model.pth"))['model_state_dict'], strict=False)
    model.eval()

    seg, seg_affine, seg_header = readnii(examples)
    results = []
    volume = []
    for i in range(0, 3):
        with torch.no_grad():
            img = val_ds[i]["image"]
            val_inputs = torch.unsqueeze(img, 1).cuda()
            val_outputs = sliding_window_inference(
                val_inputs, spatial_size, 4, model, overlap=0.8
            )
            results.append(np.array(torch.argmax(
                val_outputs, dim=1).detach().cpu()[0, :, :, :]))
            if(len(volume) == 0):
                volume = np.array(img[0, :, :, :])
            else:
                volume = np.concatenate(
                    (volume, np.array(img[0, :, :, :])), axis=2)
    results = np.array(results)

    output = np.concatenate((results[0], results[1], results[2]), axis=2)

    unique, counts = np.unique(output, return_counts=True)
    score = dict(zip(unique, counts))
    results_test = 'CTA Normal'
    try:
        if score[9] > threshold:
            results_test = 'CTA Aneurysm'
    except:
        pass
    try:
        if score[10] > threshold:
            results_test = 'CTA Cerebral Artery Thrombus'
    except:
        pass
    outputResults.append(results_test)

    seg_nifti_file = nib.Nifti1Image(output, seg_affine, seg_header)
    save = f'{file_save_path}/results/{id_files}-seg.nii.gz'
    nib.save(seg_nifti_file, save)
    outputResults.append(save)

    volum_sample_path = f'{model_dir}/1.nii.gz'
    volum, affine, header = readnii(volum_sample_path)
    volum_nifti_file = nib.Nifti1Image(volume, affine, header)
    save_volum = f'{file_save_path}/results/{id_files}-volume.nii.gz'

    nib.save(volum_nifti_file, save_volum)
    outputResults.append(save_volum)

    time_modal = time.time() - time_modal
    print('===============Time modal==================')
    print(time_modal)
    print('---------------------------------------------')

    return outputResults


def render(img_dir, seg_dir, files):
    import os
    import cv2

    import nibabel as nib
    import numpy as np
    from multiprocessing import Process

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    from tqdm import tqdm

    os.makedirs(f'{file_save_path}/render/{files}/', exist_ok=True)
    img_data = nib.load(img_dir).get_fdata()
    seg_data = nib.load(seg_dir).get_fdata()

    def rerange_img_axial(img_data):
        output = []
        for i in range(img_data.shape[2]):
            range_img = np.unique(img_data[:, :, i])
            max_new = max(range_img)
            min_new = min(range_img)
            hieu = max_new - min_new
            temp = (img_data[:, :, i]-min_new)*(255/hieu)
            output.append(temp)
        return output
    
    def rerange_img_coronal(img_data):
        output = []
        for i in range(img_data.shape[0]):
            range_img = np.unique(img_data[i, :, :])
            max_new = max(range_img)
            min_new = min(range_img)
            hieu = max_new - min_new
            temp = (img_data[i, :, :]-min_new)*(255/hieu)
            output.append(temp)
        return output
    def rerange_img_sagittal(img_data):
        output = []
        for i in range(img_data.shape[1]):
            range_img = np.unique(img_data[:, i, :])
            max_new = max(range_img)
            min_new = min(range_img)
            hieu = max_new - min_new
            temp = (img_data[:, i, :]-min_new)*(255/hieu)
            output.append(temp)
        return output
    def merge_img_seg_axial(img_data, seg_data):
        status = "done"
        os.makedirs(f'{file_save_path}/render/{files}/axial/', exist_ok=True)
        temp_img = rerange_img_axial(img_data)
        for i in range(img_data.shape[2]):
            temp_img_3 = np.zeros((128, 160, 3), np.uint8)
            temp_img_3[:, :, 0] = temp_img[i]
            temp_img_3[:, :, 1] = temp_img[i]
            temp_img_3[:, :, 2] = temp_img[i]
            temp_img_3[seg_data[:, :, i] == 1] = (255, 51, 51)
            temp_img_3[seg_data[:, :, i] == 2] = (255, 153, 51)
            temp_img_3[seg_data[:, :, i] == 3] = (255, 255, 51)
            temp_img_3[seg_data[:, :, i] == 4] = (153, 255, 51)
            temp_img_3[seg_data[:, :, i] == 5] = (51, 255, 51)
            temp_img_3[seg_data[:, :, i] == 6] = (51, 255, 153)
            temp_img_3[seg_data[:, :, i] == 7] = (51, 153, 255)
            temp_img_3[seg_data[:, :, i] == 8] = (255, 102, 153)
            temp_img_3[seg_data[:, :, i] == 9] = (51, 102, 255)
            temp_img_3 = cv2.rotate(temp_img_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(
                f'{file_save_path}/render/{files}/axial/{i}.jpg', temp_img_3)
            outputResults = [status, f'{file_save_path}/render/{files}/axial/']
        return outputResults

    def merge_img_seg_coronal(img_data, seg_data):
        status = "done"
        os.makedirs(f'{file_save_path}/render/{files}/coronal/', exist_ok=True)
        temp_img = rerange_img_coronal(img_data)
        for i in range(img_data.shape[0]):
            temp_img_3 = np.zeros((160, 528, 3), np.uint8)
            temp_img_3[:, :, 0] = temp_img[i]
            temp_img_3[:, :, 1] = temp_img[i]
            temp_img_3[:, :, 2] = temp_img[i]
            temp_img_3[seg_data[i, :, :] == 1] = (255, 51, 51)
            temp_img_3[seg_data[i, :, :] == 2] = (255, 153, 51)
            temp_img_3[seg_data[i, :, :] == 3] = (255, 255, 51)
            temp_img_3[seg_data[i, :, :] == 4] = (153, 255, 51)
            temp_img_3[seg_data[i, :, :] == 5] = (51, 255, 51)
            temp_img_3[seg_data[i, :, :] == 6] = (51, 255, 153)
            temp_img_3[seg_data[i, :, :] == 7] = (51, 153, 255)
            temp_img_3[seg_data[i, :, :] == 8] = (255, 102, 153)
            temp_img_3[seg_data[i, :, :] == 9] = (51, 102, 255)
            temp_img_3 = cv2.rotate(temp_img_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(
                f'{file_save_path}/render/{files}/coronal/{i}.jpg', temp_img_3)
            outputResults = [
                status, f'{file_save_path}/coronal/{files}/axial/']
        return outputResults

    def merge_img_seg_sagittal(img_data, seg_data):
        status = "done"
        os.makedirs(
            f'{file_save_path}/render/{files}/sagittal/', exist_ok=True)
        temp_img = rerange_img_sagittal(img_data)
        for i in range(img_data.shape[1]):
            temp_img_3 = np.zeros((128, 528, 3), np.uint8)
            temp_img_3[:, :, 0] = temp_img[i]
            temp_img_3[:, :, 1] = temp_img[i]
            temp_img_3[:, :, 2] = temp_img[i]
            temp_img_3[seg_data[:, i, :] == 1] = (255, 51, 51)
            temp_img_3[seg_data[:, i, :] == 2] = (255, 153, 51)
            temp_img_3[seg_data[:, i, :] == 3] = (255, 255, 51)
            temp_img_3[seg_data[:, i, :] == 4] = (153, 255, 51)
            temp_img_3[seg_data[:, i, :] == 5] = (51, 255, 51)
            temp_img_3[seg_data[:, i, :] == 6] = (51, 255, 153)
            temp_img_3[seg_data[:, i, :] == 7] = (51, 153, 255)
            temp_img_3[seg_data[:, i, :] == 8] = (255, 102, 153)
            temp_img_3[seg_data[:, i, :] == 9] = (51, 102, 255)
            temp_img_3 = cv2.rotate(temp_img_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(
                f'{file_save_path}/render/{files}/sagittal/{i}.jpg', temp_img_3)
            outputResults = [
                status, f'{file_save_path}/sagittal/{files}/axial/']
        return outputResults

    class MultiProcess(object):
        def __init__(self, num_process=4, num_op=100):
            self.num_process = num_process
            self.num_op = num_op
            assert self.num_process > 0
            assert self.num_op > 0
            pass

        def __call__(self):
            process_list = []
            # for _ in range(self.num_process):
            axial = Process(target=merge_img_seg_axial, args=(
                img_data, seg_data))
            axial.start()
            coronal = Process(target=merge_img_seg_coronal, args=(
                img_data, seg_data))
            coronal.start()
            sagittal = Process(target=merge_img_seg_sagittal, args=(
                img_data, seg_data))
            sagittal.start()
            process_list = [axial, coronal, sagittal]
                # process_list.append(axial, coronal, sagittal)

            for _ in range(len(process_list)):
                p = process_list[_]
                p.join()
            pass

    def processMultiple(num_cpus):
        multi = MultiProcess(num_process=num_cpus, num_op=1)
        multi()

    return processMultiple(1)


def predict(files):
    output = detect(files)
    idFile = id(files)
    render(output[1], output[2], idFile)
    status = True
    dirOutput = [        
        f'{file_save_path}/render/{idFile}/axial/',
        f'{file_save_path}/render/{idFile}/coronal/',
        f'{file_save_path}/render/{idFile}/sagittal/'
        ]
    return status,output[0],dirOutput




# def create_gif_for_volume(input_image, title='.gif', filename='test.gif'):
#     # see example from matplotlib documentation
#     import imageio
#     import matplotlib.animation as animate
#     images = []
#     input_image_data = input_image.get_fdata()
#     fig = plt.figure()
#     for i in range(input_image_data.shape[2]):
#         im = plt.imshow(input_image_data[:,:,i], cmap="gray", animated=True)
#         images.append([im])

#     ani = animate.ArtistAnimation(fig, images, interval=25,\
#         blit=True, repeat_delay=500)
#     plt.title(title, fontsize=20)
#     plt.axis('off')
#     ani.save(filename)
#     return filename

# def create_gif_for_segment(input_image, title='.gif', filename='test.gif'):
#     # see example from matplotlib documentation
#     import imageio
#     import matplotlib.animation as animate
#     images = []
#     input_image_data = input_image.get_fdata()
#     fig = plt.figure()
#     for i in range(input_image_data.shape[2]):
#         im = plt.imshow(input_image_data[:,:,i], animated=True)
#         images.append([im])

#     ani = animate.ArtistAnimation(fig, images, interval=25,\
#         blit=True, repeat_delay=500)
#     plt.title(title, fontsize=20)
#     plt.axis('off')
#     ani.save(filename)
#     return filename

import SimpleITK as sitk
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import zipfile

img_size = (288, 288)
t1_d = {}
t2f_d = {}
t1_mask = {}
t2f_mask = {}
image_shapes = []

def padding_image_array_size(image_array, out_size):
    img_z, img_x, img_y = image_array.shape[0], image_array.shape[1], image_array.shape[2]
    out_z, out_x, out_y = out_size[0], out_size[1], out_size[2]
    if out_z > img_z:
        z_up = int((out_z - img_z) / 2)
        z_down = out_z - img_z - z_up

        if out_x >= img_x and out_y >= img_y:  # 三个维度都是padding
            x_up = int((out_x - img_x) / 2)
            x_down = out_x - img_x - x_up
            y_up = int((out_y - img_y) / 2)
            y_down = out_y - img_y - y_up
            new_volume = np.pad(image_array, ((z_up, z_down), (x_up, x_down), (y_up, y_down)), mode='constant')
        else:
            new_volume = np.pad(image_array, (z_up, z_down), mode='constant')
            new_volume = img_center_crop(new_volume, (24, 256, 256))
    else:
        # 把z轴crop为32
        z_start = int((out_z - img_z) / 2)
        image_array = image_array[z_start: z_start + out_size[0], :, :]
        if out_x >= img_x and out_y >= img_y:  # 三个维度都是padding
            x_up = int((out_x - img_x) / 2)
            x_down = out_x - img_x - x_up
            y_up = int((out_y - img_y) / 2)
            y_down = out_y - img_y - y_up
            new_volume = np.pad(image_array, ((0, 0), (x_up, x_down), (y_up, y_down)), mode='constant')
        else:
            new_volume = img_center_crop(image_array, (24, 256, 256))

    return new_volume


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def _get_affine(img):
    """
    Get or construct the affine matrix of the image, it can be used to correct
    spacing, orientation or execute spatial transforms.
    Construct Affine matrix based on direction, spacing, origin information.
    Refer to: https://github.com/RSIP-Vision/medio

    Args:
        img: a ITK image object loaded from a image file.

    """
    # print(img.GetDirection())
    direction = img.GetDirection()
    spacing = np.asarray(img.GetSpacing())
    origin = np.asarray(img.GetOrigin())

    direction = np.asarray(direction)
    affine = np.eye(direction.shape[0] + 1)
    affine[(slice(-1), slice(-1))] = direction @ np.diag(spacing)
    affine[(slice(-1), -1)] = origin
    return affine


def img_center_crop(image, crop_size):
    assert len(image.shape) == 3, 'invalid image size in sliding window'
    cropping = []
    z_start, x_start, y_start = 0, 0, 0
    img_z, img_x, img_y = image.shape[0], image.shape[1], image.shape[2]
    crop_z, crop_x, crop_y = crop_size[0], crop_size[1], crop_size[2]
    # x or y 一个比crop大 或者两个都大
    if img_x > crop_x and img_y > crop_y:
        starting = [int((crop_z - img_z) / 2), int((img_x - crop_x) / 2), int((img_y - crop_y) / 2)]
        z_start, x_start, y_start = starting[0], starting[1], starting[2]

    elif img_x > crop_x and img_y <= crop_y:
        starting = [int((crop_z - img_z) / 2), int((img_x - crop_x) / 2), int((crop_y - img_y) / 2)]
        z_start, x_start, y_start = starting[0], starting[1], 0
        y_up = int((crop_y - img_y) / 2)
        y_down = crop_y - img_y - y_up
        image = np.pad(image, ((0, 0), (0, 0), (y_up, y_down)), mode='constant')

    elif img_x <= crop_x and img_y > crop_y:
        starting = [int((crop_z - img_z) / 2), int((crop_x - img_x) / 2), int((img_y - crop_y) / 2)]
        z_start, x_start, y_start = starting[0], 0, starting[2]
        x_up = int((crop_x - img_x) / 2)
        x_down = crop_x - img_x - x_up
        image = np.pad(image, ((0, 0), (x_up, x_down), (0, 0)), mode='constant')

    img_crop = image[z_start: z_start + crop_size[0], x_start:x_start + crop_size[1],
               y_start: y_start + crop_size[2]]

    return img_crop


def resample_image_array_size(image_array, out_size, order=3):
    #Bilinear interpolation would be order=1,
    # nearest is order=0,
    # and cubic is the default (order=3).
    real_resize = np.array(out_size) / image_array.shape

    new_volume = ndimage.zoom(image_array, zoom=real_resize, order=order)
    return new_volume


def read_data(dir):
    index = 0
    for each_patient_dir in os.listdir(dir):
        if dir[-1] != "/":
            dir += "/"
        if each_patient_dir[0] == ".":
            continue

        patient_path = dir + each_patient_dir
        if not os.path.isdir(patient_path):
            continue
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(patient_path)
        # print(series_ids)
        # 拿到了序列id以后呢
        t1_series = ""
        fl_series = ""
        nrrd_t1_data = ""
        nrrd_fl_data = ""
        for each_path in os.listdir(patient_path):
            if patient_path[-1] != "/":
                patient_path += "/"
            # print(each_path)
            if os.path.isdir(patient_path + each_path) and each_path[0] != ".":
                # 说明找到那个文件夹了
                dir1 = patient_path + each_path + "/"
                print(dir1)
                for series_id_dir in os.listdir(dir1):
                    if os.path.exists(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd.zip") and not  os.path.exists(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd"):
                        # 先进行解压
                        zFile = zipfile.ZipFile(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd.zip", "r")
                        for fileM in zFile.namelist():
                            zFile.extract(fileM, dir1 + series_id_dir + "/")
                        zFile.close()
                        # 说明找到了t1 序列
                        t1_series = series_id_dir
                        nrrd_t1_data = sitk.ReadImage(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd")
                        print("找到了t1序列label" + str(sitk.GetArrayFromImage(nrrd_t1_data).shape))
                        # nrrd_t1_data, nrrd_options = nrrd.read(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd")
                    elif os.path.exists(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd"):
                        # 说明找到了t1 序列
                        t1_series = series_id_dir
                        nrrd_t1_data = sitk.ReadImage(dir1 + series_id_dir + "/" + "T1WI-CE_t_1.nrrd")
                        print("找到了t1序列label" + str(sitk.GetArrayFromImage(nrrd_t1_data).shape))


                    elif os.path.exists(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd.zip") and not os.path.exists(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd"):
                        # 先进行解压
                        zFile = zipfile.ZipFile(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd.zip", "r")
                        for fileM in zFile.namelist():
                            zFile.extract(fileM, dir1 + series_id_dir + "/")
                        zFile.close()
                        # 说明找到了t2 序列
                        t1_series = series_id_dir
                        nrrd_fl_data = sitk.ReadImage(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd")
                        print("找到了t2序列label" + str(sitk.GetArrayFromImage(nrrd_fl_data).shape))

                    elif os.path.exists(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd"):
                        # 找到了flare 序列
                        fl_series = series_id_dir
                        nrrd_fl_data = sitk.ReadImage(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd")
                        print("找到了t2序列label" + str(sitk.GetArrayFromImage(nrrd_fl_data).shape))
                        # nrrd_fl_data, nrrd_options = nrrd.read(dir1 + series_id_dir + "/" + "FL-CE_e_1.nrrd")


        # print(nrrd_options)
        ## 序列的话，看了下 一共五个序列，但是应该是只用1号和4号 这两个序列
        dicom_series_t1 = reader.GetGDCMSeriesFileNames(patient_path, t1_series)
        # file = sitk.ReadImage(patient_path)

        reader.SetFileNames(dicom_series_t1)
        img_t1 = reader.Execute()

        img_array_t1 = sitk.GetArrayFromImage(img_t1)

        # print("t1 series: " + str(img_array_t1.shape))

        # print("t1 label is {}".format(nrrd_t1_data.shape))

        dicom_series_fl = reader.GetGDCMSeriesFileNames(patient_path, fl_series)
        reader.SetFileNames(dicom_series_fl)
        img_fl = reader.Execute()
        space_fl = img_fl.GetSpacing()
        img_array_fl = sitk.GetArrayFromImage(img_fl)

        if nrrd_t1_data == "":
            print("此人无t1序列 跳过")
            continue

        # print(nrrd_t1_data.shape)
        # num = len(img_array_t1)
        # num_fl = len(img_array_fl)
        # num = min(num, num_fl)
        # if "CHEN_SHA_LIN" in dir1:
        # img_array_t1 = resample_image_array_size(img_array_t1, out_size=(32, 256, 256), order=3)
        # img_array_fl = resample_image_array_size(img_array_fl, out_size=(32, 256, 256), order=3)
        # nrrd_t1_data = resample_image_array_size(nrrd_t1_data, out_size=(32, 256, 256), order=1)
        # nrrd_fl_data = resample_image_array_size(nrrd_fl_data, out_size=(32, 256, 256), order=1)

        # print(np.unique(nrrd_t1_data))
        # for k in range(2, 30):
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(img_array_t1[k], cmap="gray")
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(nrrd_t1_data[k], cmap="gray")
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(img_array_fl[k], cmap="gray")
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(nrrd_fl_data[k], cmap="gray")
        #     plt.show()
        # print("t2_flare series: " + str(img_array_fl.shape))
        # print("fl label is {}".format(nrrd_fl_data.shape))

       
            
        # os._exit(0)
        resampled_image = resample_image(img_t1, (1., 1., 6.5))  # itk_image.GetSize (x,y,z)
        resampled_image = sitk.GetArrayFromImage(resampled_image)  # GetArrayFromImage (z,x,y)
        image_resample_t1 = padding_image_array_size(resampled_image, out_size=(24, 256, 256))
       

        # print(each_patient_dir)
        # print(image_resample_t1.shape)
        resampled_image = resample_image(img_fl, (1., 1., 6.5))  # itk_image.GetSize (x,y,z)
        resampled_image = sitk.GetArrayFromImage(resampled_image)  # GetArrayFromImage (z,x,y)
        image_resample_t2 = padding_image_array_size(resampled_image, out_size=(24, 256, 256))
       
        # print(image_resample_t2.shape)

        resampled_image = resample_image(nrrd_t1_data, (1., 1., 6.5), is_label=True)  # itk_image.GetSize (x,y,z)
        resampled_image = sitk.GetArrayFromImage(resampled_image)  # GetArrayFromImage (z,x,y)
        image_resample_t1_label = padding_image_array_size(resampled_image, out_size=(24, 256, 256))
       
        if nrrd_fl_data == "":
            # 如果没有水肿区域，则全0初始化即可。
            image_resample_t2_label = np.zeros_like(image_resample_t2)
            # itk_image_resample = sitk.GetImageFromArray(nrrd_fl_data)
            # sitk.WriteImage(itk_image_resample, "./data/label_data/mask/" + each_patient_dir + '_t2_mask.nii.gz')
        else:
            resampled_image = resample_image(nrrd_fl_data, (1., 1., 6.5), is_label=True)  # itk_image.GetSize (x,y,z)
            resampled_image = sitk.GetArrayFromImage(resampled_image)  # GetArrayFromImage (z,x,y)
            image_resample_t2_label = padding_image_array_size(resampled_image, out_size=(24, 256, 256))
           
       

        image_resample_t1 = resample_image_array_size(image_resample_t1, out_size=(32, 256, 256), order=3)
        image_resample_t2 = resample_image_array_size(image_resample_t2, out_size=(32, 256, 256), order=3)
        image_resample_t1_label = resample_image_array_size(image_resample_t1_label, out_size=(32, 256, 256), order=1)
        image_resample_t2_label = resample_image_array_size(image_resample_t2_label, out_size=(32, 256, 256), order=1)


        image_resample = np.stack([image_resample_t1, image_resample_t2])
        image_resample_label = np.stack([image_resample_t1_label, image_resample_t2_label])

        # for k in range(5, 28):
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(image_resample_t1[k], cmap="gray")
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(image_resample_t1_label[k], cmap="gray")
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(image_resample_t2[k], cmap="gray")
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(image_resample_t2_label[k], cmap="gray")
        #     plt.show()
        #     break


        h5_file_img = h5py.File("./data/Meningiomas/" + each_patient_dir + "_data.h5", "w")
        h5_file_img.create_dataset("image", data=image_resample, compression="gzip")
        h5_file_img.create_dataset("label", data=image_resample_label, compression="gzip")
        h5_file_img.close()


if __name__ == "__main__":
    ## 处理原始数据
    # read_data("./Grade I(所有病人数据)/")
    # read_data("./data/label/Grade_1")
    # read_data("/home/datasets/Meningiomas/Data_Processing/label/Grade_1/")

    # read_data("/home/datasets/Meningiomas/Data_Processing/label/Grade_1/")
    read_data("/home/datasets/Meningiomas/Data_Processing/label/Grade_2_invasion/")
    read_data("/home/datasets/Meningiomas/Data_Processing/label/Grade_2_noninvasion/")



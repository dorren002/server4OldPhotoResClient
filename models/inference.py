import os, cv2
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms

from models.restoration_models import networks

def __make_power_2(img, base, method=Image.Resampling.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def get_transform(normalize=True, method=Image.Resampling.BICUBIC):
    transform_list = []
    
    base = float(2 ** 4)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def tensor2im(image_tensor, imtype=np.uint8, normalize=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

class RestorationModel():
    def __init__(self, root="tmp", input_dir="input", mask_dir="mask", output_dir="output"):
        self.in_dir = os.path.join(root, input_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.out_dir = os.path.join(root, output_dir)
        self.ckpt_dir = "models/checkpoints/ffc.pth"
        self.transforms = get_transform(normalize=False)
    
    def inference(self, filename=None):
        self.model = networks.define_G(3, 3, 64, "ffc", 4, 9, 1, 3, "instance", [0], "models/restoration_models/options.yaml")
        self.model.load_state_dict(torch.load(self.ckpt_dir))    

        if filename==None:
            self.batchProc()
        else:
            self.pieceProc(filename)

    def batchProc(self):
        imagelist = os.listdir(self.in_dir)
        imagelist.sort()

        idx = 0
        for image_name in imagelist:
            idx += 1
            print("processing", image_name)
            self.pieceProc(image_name)
        del self.model
            
    def pieceProc(self, filename):
        # Load Image
        scratch_file = os.path.join(self.in_dir, filename)
        mask_file = os.path.join(self.mask_dir, filename)
        
        if not os.path.isfile(scratch_file):
            print("Skipping non-file %s" % filename)
            return 
        old_image = Image.open(scratch_file).convert("RGB")
        w, h = old_image.size

        # PIL -> Tensor
        # transformed_image_PIL = data_transforms(old_image, config.input_size)
        old_image = self.transforms(old_image.convert('RGB'))

        # mask = np.array(Image.open(mask_file)).astype(np.uint8)/255
        # mask = mask.transpose(2,0,1)
        # old_image *= (1-mask)
        
        old_image = old_image.float().to(0)

        generated = self.model(old_image.unsqueeze(0))

        self.save(tensor2im(generated.data[0]), filename)
    
    def save(self, output, filename):
        cv2.imwrite(os.path.join(self.out_dir, filename), output[...,::-1])


if __name__ == "__main__" :
    rmHelper = RestorationModel()
    rmHelper.inference("real_1.png")

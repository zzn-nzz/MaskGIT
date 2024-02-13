import torch
import numpy as np
from configs import FLAGS
from models.transformer import MaskGIT
from datasets import TinyImageNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from vqvae_main import imshow
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from models.transformer import MaskGIT
from models.vqvae import VQVAE
import cv2

class Inpainting:
    def __init__(self, FLAGS):
        if FLAGS.device < 0:
           self.device = torch.device('cpu')
        else:
           self.device = torch.device(f'cuda:{FLAGS.device}')
           
        self.FLAGS = FLAGS
        self.model = MaskGIT(FLAGS).to(self.device)
        self.model.vqvae = VQVAE(num_embeddings=FLAGS.num_embeddings,
                           embedding_dim=FLAGS.embedding_dim,
                           lamda=FLAGS.lamda).to(self.device)
    
    def load_model(self):
        self.model.load_state_dict(torch.load(FLAGS.transformer_path, map_location=self.device))

    def image_completion_encode(self, incomplete_image):
        self.model.eval()
        with torch.no_grad():
            # encode the incomplete image with vq_vae encoder
            input = incomplete_image.to(self.device)
            input = input.unsqueeze(0)
            # print('input的形状：', input.shape)
            _, indices = self.model.encode(input)
            # print('indices的形状：', indices.shape)
            batch_size = input.shape[0]
            return indices, batch_size
        
    def image_completion_decode(self, indices, mask, batch_size, pred_iters = 20):
        self.model.eval()
        with torch.no_grad():
            # print('初始indices:', indices)
            masked = self.model.mask_id * torch.ones_like(indices, device=indices.device)
            masked_indices = mask * indices + (~mask) * masked

            # create indeces to be sent to transformer
            start = self.model.sos_id * torch.ones(batch_size, 1, dtype=torch.long, device=indices.device)
            masked_indices = torch.cat((start, masked_indices), dim=1)
            target_indices = torch.cat((start, indices), dim=1)

            # predict with transformer
            
            logits = self.model.transformer(masked_indices)
            mask_num = mask.shape[1] - mask.sum(dim=1)
            # print(mask_num)
            # assert False
            res_indices = target_indices

            for iter in range(pred_iters):
                t = iter + 1
                this_pred = mask.shape[1] - int(mask_num * (self.model.gamma(t/pred_iters)))
                if iter > 0:
                    this_pred -= mask.shape[1] - int(mask_num * (self.model.gamma((t - 1)/pred_iters)))
                # print(this_pred)
                    
                logits = self.model.transformer(masked_indices)
                pred_prob, pred_indices = logits.max(dim=-1)
                # print('pred_prob的形状：', pred_prob.shape)
                _, top_indices = pred_prob.topk(pred_prob.shape[1], dim=1)
                # print('top_indices:',top_indices.shape)
                for i in range(mask.shape[0]):
                    finish_pred = 0
                    for m in range(top_indices.shape[1]):
                        j = top_indices[i, m]
                        # print(j)
                        # 是开始标记
                        if j == 0:
                            continue
                        # 是一个没有被预测的
                        if mask[i, j - 1] == 0:
                            mask[i, j - 1] = 1
                            res_indices[i, j] = pred_indices[i, j]
                            finish_pred += 1
                        if finish_pred >= this_pred:
                            break

            res_indices = res_indices[..., 1:]
            res_indices = res_indices.reshape(res_indices.shape[0], 16, 16)
            # print('预测后indices：', res_indices)
            res_vector = self.model.vqvae.vq_layer.embeddings[:, res_indices]
            # print(res_vector.shape)
            res_vector = res_vector.permute(1, 0, 2, 3)
            '''
            print(res_indices)
            print('embedding形状', self.model.vqvae.vq_layer.embeddings.shape)
            res_vector = self.model.vqvae.vq_layer.embeddings[res_indices]
            print(res_vector)
            res_vector = res_vector.view(16, 16, self.model.vqvae.embedding_dim)
            res_vector = res_vector.permute(2, 0, 1)
            '''
            # decode the image
            res_image = self.model.vqvae.vqdecoder(res_vector)
            res_image = res_image.squeeze(0)
            return res_image.cpu()

    def inpainting(self, incomplete_image, left_up, right_down):
        indices, batch_size = self.image_completion_encode(incomplete_image)

        # mask==1: known place; mask==0: unkown place
        mask = torch.ones(indices.shape, dtype = torch.bool, device=indices.device)
        # print('mask的形状:', mask.shape)

        for y in range(left_up[1], right_down[1] + 1):
            for x in range(left_up[0], right_down[0] + 1):
                i = y // 16
                j = x // 16
                index = i * 16 + j
                mask[..., index] = 0

        res = self.image_completion_decode(indices=indices, mask=mask, batch_size=batch_size)
        return res
    
    def outpainting(self, incompltet_image, left_up, right_down):
        indices, batch_size = self.image_completion_encode(incompltet_image)

        # convert to 16*16
        mask = torch.zeros(indices.shape, dtype = torch.bool, device=indices.device)

        for y in range(left_up[1], right_down[1] + 1):
            for x in range(left_up[0], right_down[0] + 1):
                i = y // 16
                j = x // 16
                index = i * 16 + j
                mask[..., index] = 1

        res = self.image_completion_decode(indices=indices, mask=mask, batch_size=batch_size)
        return res
        
def load_and_process_image(image_path, left_up_old, right_down_old):
    # Load the image
    image = Image.open(image_path)

    # Get the original image size
    original_width, original_height = image.size

    # Resize to 256*256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image)

    # Calculate the new coordinates
    width_ratio = 256 / original_width
    height_ratio = 256 / original_height
    left_up_new = (int(left_up_old[0] * width_ratio), int(left_up_old[1] * height_ratio))
    right_down_new = (int(right_down_old[0] * width_ratio), int(right_down_old[1] * height_ratio))

    return image, left_up_new, right_down_new

def save_image(tensor, filename):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image_rgb = image.convert("RGB")
    image_rgb.save(filename)
    '''
    # img_cpu = img_cpu / 2 + 0.5
    npimg = tensor.numpy()
    
    # Convert from CHW to HWC
    npimg = np.transpose(npimg, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255] and ensure it's in uint8 format
    npimg = (npimg * 255).astype(np.uint8)

    # Save the image using OpenCV
    cv2.imwrite(filename, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))
    '''

if __name__ == '__main__':
    image_path = 'bucket_mask.jpg'
    output_path = 'output_bucket.png'
    left_up = (0, 499)
    right_down = (599, 999)
    
    image, left_up_new, right_down_new = load_and_process_image(image_path=image_path, left_up_old=left_up, right_down_old=right_down)

    solver = Inpainting(FLAGS)

    solver.load_model()

    res = solver.inpainting(image, left_up_new, right_down_new)

    save_image(res, output_path)

        


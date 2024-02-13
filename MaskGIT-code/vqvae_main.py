import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from models import VQVAE
from torchvision import transforms
from datasets import TinyImageNet
from configs import FLAGS
from tqdm import tqdm
import torchvision
import cv2
from torch import nn
import os


import matplotlib.pyplot as plt

class VQVAESolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS          # config

        self.start_epoch = 1
        self.model = None          
        self.optimizer = None     
        self.scheduler = None      
        if FLAGS.run == "train":
            self.summary_writer = SummaryWriter(self.FLAGS.logdir) 
        self.train_loader = None   
        self.test_loader = None    

        # choose device for train or test 
        if FLAGS.device < 0:       
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{FLAGS.device}')
        # ......

    def config_model(self):
        self.model = VQVAE(num_embeddings=FLAGS.num_embeddings,
                           embedding_dim=FLAGS.embedding_dim,
                           lamda=FLAGS.lamda)
        if FLAGS.num_devices > 1:
            print('using multiple gpus!')
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def get_dataset(self, flag):
        print(self.FLAGS)
        dataset = None
        if flag == 'test':
            dataset = TinyImageNet(self.FLAGS.root_dir, self.FLAGS.filelist, False, False)
        else:
            dataset = TinyImageNet(self.FLAGS.root_dir, self.FLAGS.filelist, True, True)
        return dataset
    
    def config_dataloader(self, disable_train=False):
        if not disable_train:
            self.train_loader = DataLoader(dataset=self.get_dataset('train'),
                                           batch_size=FLAGS.batch_size,
                                           shuffle=True)
                                           # num_workers=FLAGS.num_workers)
            
        self.test_loader = DataLoader(dataset=self.get_dataset('test'),
                                      batch_size=FLAGS.batch_size,
                                      shuffle=False)
                                      # num_workers=FLAGS.num_workers

    def config_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=FLAGS.learning_rate, betas=[0.9, 0.96])

    def config_scheduler(self):
        self.scheduler = StepLR(self.optimizer, step_size=FLAGS.lr_step_size, gamma=FLAGS.lr_gamma)

    def train(self):
        print(self.FLAGS.logdir + '/vqvae_model_tiny_imgnet.pth')
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        # set model as train mode
        # read data
        # model forward process
        # compute loss
        # compute gradient
        # optimize parameters
        # ......
        batches = 0
        for epoch in range(self.start_epoch, self.FLAGS.max_epochs + 1):
            self.model.train()  # Set the model to train mode
            for data in tqdm(self.train_loader, desc=f'batch', unit='batch'):
                inputs = data.to(self.device)

                # Forward pass
                outputs, loss = self.model(inputs)
                #print(loss)
                # Compute gradient and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.summary_writer.add_scalar('Training Loss', loss.sum().item(), batches)
                batches += 1


            # Update learning rate
            self.scheduler.step()
            torch.save(self.model.state_dict(), self.FLAGS.logdir + f'/vqvae_improved_batch_size8_epoch{epoch}.pth')
            self.my_val(epoch)

    def test(self):
        self.config_model()
        self.config_dataloader(True)
        path = FLAGS.vqvae_path
        self.load_model_from(path)
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            num = 0
            for data in tqdm(self.test_loader, desc=f'batch', unit='batch'):
                inputs = data.to(self.device)
                # Forward pass
                outputs, loss = self.model(inputs)
                self.save_images(inputs, FLAGS.vqvae_test_dataset1, num)
                self.save_images(outputs, FLAGS.vqvae_test_dataset2, num)
                num += data.shape[0]

    def save_images(self, image_data: torch.Tensor, path, num):
        for i in range(image_data.shape[0]):
            img = image_data[i]
            img_cpu = img.cpu()
        
            # img_cpu = img_cpu / 2 + 0.5
            npimg = img_cpu.numpy()
            
            # Convert from CHW to HWC
            npimg = np.transpose(npimg, (1, 2, 0))
            
            # Convert from [0, 1] to [0, 255] and ensure it's in uint8 format
            npimg = (npimg * 255).astype(np.uint8)
            #print(save_path)
            # Save the image using OpenCV
            cv2.imwrite(os.path.join(path, f'{num + i}.png'), cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))

    def manual_seed(self):
        rand_seed = self.FLAGS.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def run(self):
        eval('self.%s()' % self.FLAGS.run)

    def load_model_from(self, path='logs/maskgit/vqvae_model_tiny_imgnet.pth'):
        self.config_model()
        self.model.load_state_dict(torch.load(path))
    
    def my_val(self, epoch=0):
        pre = 'vqvae_visual' + str(epoch) + '_'
        path = '/root/autodl-tmp/CV-Final-Project/logs/maskgit/vqvae_improved_epoch38.pth'
        self.load_model_from(path)
        self.config_dataloader()
        
        dataiter = iter(self.test_loader)
        inputs = next(dataiter).to(self.device)
        print(inputs.shape)
        with torch.no_grad():
            outputs, _ = self.model(inputs)
        print(outputs.shape)
        imshow(torchvision.utils.make_grid(inputs), pre, 1)
        #print(4)
        imshow(torchvision.utils.make_grid(outputs), pre, 2)

    def load_model(self):
        weight = 'logs/maskgit/vqvae_model_tiny_imgnet_trained.pth'
        self.config_model()
        self.model.load_state_dict(torch.load(weight))
        return self.model

    def see_codebook(self):
        path = 'vqvae_resized_epoch20.pth'
        self.load_model_from(path)
        
        v = torch.zeros(512)
        v[15] = 1.0
        vs = [v for _ in range(64 * 64)]
        vs = torch.cat(vs)
        vs = vs.view(64, 64, 512)
        vs = vs.permute(2, 0, 1)
        vs = torch.unsqueeze(vs, 0)
        vs = vs.to(self.device)
        print(vs.shape)
        with torch.no_grad():
            outputs = self.model.decoder(vs)
        pre = "cb_"
        imshow(torchvision.utils.make_grid(outputs), pre, 2)
                

    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        completion.run()
        #completion.my_val()
        #completion.see_codebook()

def imshow(img, pre, num):
    save_path = f'{pre}_{num}.png'
    img_cpu = img.cpu()
    
    # img_cpu = img_cpu / 2 + 0.5
    npimg = img_cpu.numpy()
    
    # Convert from CHW to HWC
    npimg = np.transpose(npimg, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255] and ensure it's in uint8 format
    npimg = (npimg * 255).astype(np.uint8)
    #print(save_path)
    # Save the image using OpenCV
    cv2.imwrite(save_path, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    VQVAESolver.main()
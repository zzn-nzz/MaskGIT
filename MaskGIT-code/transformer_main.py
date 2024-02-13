import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from configs import FLAGS
from models.transformer import MaskGIT
from torch.optim.lr_scheduler import StepLR
from datasets import TinyImageNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.vqvae import VQVAE
from vqvae_main import imshow
import torchvision
import os
from lr_schedule import WarmupLinearLRSchedule
import cv2

class TransformerSolver:
    def __init__(self, FLAGS):
        if FLAGS.device < 0:
           self.device = torch.device('cpu')
        else:
           self.device = torch.device(f'cuda:{FLAGS.device}')
           
        self.FLAGS = FLAGS
        self.start_epoch = FLAGS.start_epoch
        self.model = MaskGIT(FLAGS).to(self.device)          # torch.nn.Module
        self.optimizer = None       # torch.optim.Optimizer  
        self.scheduler = None       # torch.optim.lr_scheduler._LRScheduler
        self.train_loader = None
        self.test_loader = None
        if not os.path.exists(self.FLAGS.logdir):
            os.mkdir(FLAGS.logdir)
        if FLAGS.run == "train":
            self.summary_writer = SummaryWriter(self.FLAGS.logdir) 
        # .....
            if self.start_epoch > 1:
                self.load_ckpt(self.start_epoch - 1)

    def load_ckpt(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(f"logs/transformer/6/bitrans_tiny_imgnet_epoch{epoch}.pth")))
        print("ckpt loaded")

    def load_vqvae(self):
        self.model.vqvae = VQVAE(num_embeddings=FLAGS.num_embeddings,
                           embedding_dim=FLAGS.embedding_dim,
                           lamda=FLAGS.lamda).to(self.device)
        self.model.vqvae.load_state_dict(torch.load(FLAGS.vqvae_path))

    def get_dataset(self, flag):
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
        self.optimizer = torch.optim.Adam(self.model.transformer.parameters(), 
                                          lr = 1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)

    def config_scheduler(self):
        self.scheduler = WarmupLinearLRSchedule(
            optimizer=self.optimizer,
            init_lr=1e-6,
            peak_lr=FLAGS.learning_rate,
            end_lr=0.,
            warmup_epochs=10,
            epochs=FLAGS.max_epochs,
            current_step=1
        )
        
    def config_log(self):
        log_path = 'logs/config/Maskgit.txt'
        with open(log_path, 'a') as file:
            file.write(str(self.FLAGS))

    def train(self):
        self.manual_seed()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        self.config_log()
        self.load_vqvae()
        #self.my_val()
        #quit()
        batches = 0
        #print(self.model.vqvae.vqdecoder())
        for epoch in range(self.start_epoch, self.FLAGS.max_epochs + 1):
            print(f"Epoch {epoch} starts:")
            self.model.train()  # Set the model to train mode
            for data in tqdm(self.train_loader, desc=f'batch', unit='batch'):
                inputs = data.to(self.device)
                logits, target = self.model(inputs)
                #print(logits.shape)
                #print(target.shape)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                                         label_smoothing=0.1)
                # Compute gradient and optimize
                if batches % 10 == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss.backward()
                self.summary_writer.add_scalar('Training Loss', loss.item(), batches)
                batches += 1


            # Update learning rate
            self.scheduler.step()
            torch.save(self.model.state_dict(), self.FLAGS.logdir + f'/bitrans_tiny_imgnet_epoch{epoch}.pth')
            try:
                self.my_val(epoch)
                print('success val')
            except:
                pass
        # set model as train mode
        # read data
        # model forward process
        # compute loss
        # compute gradient
        # optimize parameters
        # ......

    def test(self):
        self.manual_seed()
        self.config_dataloader(True)
        path = FLAGS.transformer_path
        self.model.load_state_dict(torch.load(path))
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            num = 0
            for data in tqdm(self.test_loader, desc=f'batch', unit='batch'):
                inputs = data.to(self.device)
                # Forward pass
                logits, target = self.model(inputs)
                outputs = self.logits2img(logits)
                self.save_images(inputs, FLAGS.transformer_test_dataset1, num)
                self.save_images(outputs, FLAGS.transformer_test_dataset2, num)
                num += data.shape[0]
                #quit()

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
    @torch.no_grad()
    def my_val(self, epoch=0):
        pre = "maskgit_epoch" + str(epoch)
        dataiter = iter(self.test_loader)
        inputs = next(dataiter).to(self.device)
        #print(inputs.shape)
        with torch.no_grad():
            outputs, _ = self.model(inputs)
        #print(outputs.shape)
        outputs = self.logits2img(outputs)
        imshow(torchvision.utils.make_grid(inputs), pre, 1)
        #print(4)
        imshow(torchvision.utils.make_grid(outputs), pre, 2)

    @torch.no_grad()
    def logits2img(self, outputs):
        outputs = outputs[:, 1:, :]
        outputs = outputs.argmax(dim=2)
        outputs = outputs.reshape(outputs.shape[0], 16, 16)
        features = self.model.vqvae.vq_layer.embeddings[:, outputs]
        features = features.permute(1, 0, 2, 3)
        #print(features.shape)
        outputs = self.model.vqvae.vqdecoder(features)
        return outputs


    def run(self):
        eval('self.%s()' % self.FLAGS.run)
        
    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        completion.run()

if __name__ == '__main__':
    TransformerSolver.main()

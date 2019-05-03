1). Name: Jingtao Li, Zhaofeng Zhang
    Selected topic: Media generation
    Contributions: Jingtao Li: Implement the FreezeGAN module
                   Zhaofeng Zhang: Implement the MMD metrics module

2). Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
    Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
    Borji, Ali. "Pros and cons of gan evaluation measures." Computer Vision and Image Understanding 179 (2019): 41-65.

3). https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

4). http://yann.lecun.com/exdb/mnist/index.html

5). Final_Project:
	->Revised_Code
		->FreezeGAN.py: Revised code (Input Folder is not needed)
		->NonFreeze4000: Results from DCGAN with 4000 steps (Output Folder 1)
		->NonFreeze8000: Results from DCGAN with 8000 steps (Output Folder 2)
                ->FreezeGheadDtail70_4000: Results from FreezeGAN with 4000 steps (Output Folder 3)
	->environment.yml

6). Implement the FreezeGAN module and the MMD metrics module

7). a. MMD metric can better show the traning progress and convergence status.
    b. FreezeGAN module can perform training only on the selected layers with the other freezed for better convergence.

8). Runing Instruction:

To Set up the environment, use the command:

install miniconda3 in a linux-x64 machine:

make sure cuda is installed.

conda env create -f environment.yml

conda activate keras

python FreezeGAN.py
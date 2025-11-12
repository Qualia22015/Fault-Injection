# Fault-Injection
Generate dataset from MNIST with Pytorch-Fault-Simulator by 4thMemorize

# Requirements (Lib)
  torch (*)
  torchvision (*)
  
# Tutorial
  Build CNN based on ResNet50, trained weights will saved as "mnist_cnn50_trained.pth"
  With "mnist_cnn50_trained.pth", run Fault_Injection.py

  Error injected data will saved to local directory folder named "feature traces"

  Data saved at feature traces will used at EDMN.

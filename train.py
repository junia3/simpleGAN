from torch import optim
import torch
import torch.nn as nn
import time

def train(params, dataloader, model_discriminator, model_generator, loss_function = nn.BCELoss(), device = None):

  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Current device is :", device)

  lr = params['learning rate']
  beta1 = params['beta1']
  beta2 = params['beta2']
  num_epochs = params['epoch']
  ns = params["noise"]

  optim_discriminator = optim.Adam(model_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
  optim_generator = optim.Adam(model_generator.parameters(), lr=lr, betas=(beta1, beta2))

  start_time = time.time()
  loss_history = {"generator": [], "discriminator": []}
  model_discriminator.train()
  model_generator.train()

  for epoch in range(num_epochs):
    for idx, (x, y) in enumerate(dataloader):
      batch_size = x.shape[0]

      real_image = x.to(device)
      y_real = torch.ones(batch_size, 1).to(device)
      y_fake = torch.zeros(batch_size, 1).to(device)

      optim_generator.zero_grad()
      noise = torch.randn(batch_size, ns, device=device)
      fake_image = model_generator(noise)
      check_image = model_discriminator(fake_image)

      loss_generator = loss_function(check_image, y_real)
      loss_generator.backward()
      optim_generator.step()

      optim_discriminator.zero_grad()

      out_real = model_discriminator(real_image)
      out_fake = model_discriminator(fake_image.detach())
      loss_real = loss_function(out_real, y_real)
      loss_fake = loss_function(out_fake, y_fake)
      loss_discriminator = (loss_real + loss_fake) / 2

      loss_discriminator.backward()
      optim_discriminator.step()

      loss_history["generator"].append(loss_generator.item())
      loss_history["discriminator"].append(loss_discriminator.item())

      if idx % 1000 == 0:
        print("Epoch : %.0f, G_loss : %.4f, D_loss : %.4f, time : %.2f min"
              %(epoch, loss_generator.item(), loss_discriminator.item(), (time.time() - start_time)/60))
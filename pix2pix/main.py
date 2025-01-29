import torch
from models import Generator, Discriminator
from train.train import train_gan
from data.dataloader import train_dataloader, valid_dataloader

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = Generator().to(device)  
D = Discriminator().to(device)  

G_optim = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
D_optim = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

loss1 = torch.nn.BCELoss()
loss2 = torch.nn.L1Loss()
LAMBDA = 7

best_G, best_D, best_epoch = train_gan(
    G=G,
    D=D,
    G_optim=G_optim,
    D_optim=D_optim,
    loss1=loss1,
    loss2=loss2,
    train_dataloader=train_dataloader,
    val_dataloader=valid_dataloader,
    device=device,
    LAMBDA=LAMBDA,
    max_epochs=60,
    patience=10
)

best_G_state = G.state_dict()
best_D_state = D.state_dict()

torch.save(best_G_state, 'best_G.pth')
torch.save(best_D_state, 'best_D.pth')
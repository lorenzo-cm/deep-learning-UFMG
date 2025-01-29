import os
import torch
from tqdm import tqdm
import json

from data.plot import gen_img_plot

def train_gan(
    G, 
    D, 
    G_optim, 
    D_optim, 
    loss1,
    loss2,
    train_dataloader,
    val_dataloader,
    device, 
    LAMBDA, 
    max_epochs,
    validation=True,
    patience=10,
):
    """
    GAN training
    
    G: Generator model 
    D: Discriminator model
    G_optim: optimizer for G
    D_optim: optimizer for D
    loss1: adversarial loss
    loss2: reconstruction loss
    train_dataloader: train dataloader returning img, mask 
    val_dataloader: valid dataloader returning img, mask 
    device: torch device
    LAMBD: regularization factor
    max_epochs: max training epochs
    validation: allows validation step and early stopping
    patience=10: early stopping
    """
    # Creates the dir for logging the training data
    os.makedirs('train_log', exist_ok=True)

    # History lists to store training metrics
    D_loss_history = []
    G_loss_history = []
    D_acc_history  = []

    # History lists to store validation metrics
    val_D_loss_history = []
    val_G_loss_history = []
    val_D_acc_history  = []

    # Move models to the specified device if not already
    G.to(device)
    D.to(device)

    # Early Stopping Variables
    best_val_g_loss = float('inf')
    best_epoch      = 0
    patience_count  = 0

    # Store best models
    best_G_state = None
    best_D_state = None

    ######################################################################
    # Training Phase
    ######################################################################
    for epoch in tqdm(range(max_epochs)):
        G.train()
        D.train()

        # Train metrics: losses and discriminator acc
        train_D_epoch_loss = 0.0
        train_G_epoch_loss = 0.0
        
        train_D_epoch_acc  = 0.0
        train_D_epoch_real_acc = 0.0
        train_D_epoch_fake_acc = 0.0

        train_batch_count = len(train_dataloader)

        # imgs: b/w images
        # masks: ground truth colored images
        for imgs, masks in tqdm(train_dataloader):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            #-----------------------------------------------------
            # 1) Train the Discriminator
            #-----------------------------------------------------
            D_optim.zero_grad()

            # Real pass
            D_real = D(imgs, masks) 
            D_real_loss = loss1(D_real, torch.ones_like(D_real))
            D_real_loss.backward()

            # Fake pass
            G_img    = G(imgs)
            D_fake   = D(imgs, G_img.detach())
            D_fake_loss = loss1(D_fake, torch.zeros_like(D_fake))
            D_fake_loss.backward()

            # Combine and step
            D_loss = D_real_loss + D_fake_loss
            D_optim.step()

            #-----------------------------------------------------
            # 2) Train the Generator (2 times)
            #-----------------------------------------------------
            current_G_loss = 0.0
            for _ in range(2):
                G_optim.zero_grad()

                G_img  = G(imgs)
                D_fake = D(imgs, G_img)
                
                # Adversarial loss
                G_loss_adv = loss1(D_fake, torch.ones_like(D_fake))
                
                # Reconstruction loss
                G_loss_L1  = loss2(G_img, masks)
                
                G_loss = G_loss_adv + LAMBDA * G_loss_L1
                G_loss.backward()
                G_optim.step()
                
                # Store the last loss
                current_G_loss = G_loss

            #-----------------------------------------------------
            # 3) Track Training Stats
            #-----------------------------------------------------
            train_D_epoch_loss += D_loss.item()
            train_G_epoch_loss += current_G_loss.item()

            # Discriminator accuracy
            D_real_acc = (D_real >= 0.5).float().mean().item()
            D_fake_acc = (D_fake < 0.5).float().mean().item()
            
            train_D_epoch_real_acc += D_real_acc
            train_D_epoch_fake_acc += D_fake_acc
            
            train_D_epoch_acc += (D_real_acc + D_fake_acc) / 2

        # Average over the training batches
        train_D_epoch_loss /= train_batch_count
        train_G_epoch_loss /= train_batch_count
        train_D_epoch_acc  /= train_batch_count
        
        train_D_epoch_real_acc /= train_batch_count
        train_D_epoch_fake_acc /= train_batch_count

        D_loss_history.append(train_D_epoch_loss)
        G_loss_history.append(train_G_epoch_loss)
        D_acc_history.append(train_D_epoch_acc)
        
        
        ######################################################################
        # Validation Phase
        ######################################################################
        if validation:
            
            val_D_epoch_loss = 0.0
            val_G_epoch_loss = 0.0
            val_D_epoch_acc  = 0.0
            
            G.eval()
            D.eval()

            val_batch_count = len(val_dataloader)

            with torch.no_grad():
                for imgs, masks in val_dataloader:
                    imgs  = imgs.to(device)
                    masks = masks.to(device)

                    # Discriminator on Real
                    D_real = D(imgs, masks)
                    D_real_loss = loss1(D_real, torch.ones_like(D_real))

                    # Discriminator on Fake
                    G_img  = G(imgs)
                    D_fake = D(imgs, G_img)
                    D_fake_loss = loss1(D_fake, torch.zeros_like(D_fake))

                    D_loss = D_real_loss + D_fake_loss

                    # Generator Loss
                    G_loss_adv = loss1(D_fake, torch.ones_like(D_fake))
                    G_loss_L1  = loss2(G_img, masks)
                    G_loss     = G_loss_adv + LAMBDA * G_loss_L1

                    val_D_epoch_loss += D_loss.item()
                    val_G_epoch_loss += G_loss.item()

                    # Discriminator accuracy
                    D_real_acc = (D_real >= 0.5).float().mean().item()
                    D_fake_acc = (D_fake < 0.5).float().mean().item()
                    val_D_epoch_acc += (D_real_acc + D_fake_acc) / 2

            val_D_epoch_loss /= val_batch_count
            val_G_epoch_loss /= val_batch_count
            val_D_epoch_acc  /= val_batch_count

            # Store validation stats
            val_D_loss_history.append(val_D_epoch_loss)
            val_G_loss_history.append(val_G_epoch_loss)
            val_D_acc_history.append(val_D_epoch_acc)

            # Print training and validation info
            print(
                f"[Epoch {epoch+1}/{max_epochs}] "
                f"Train - G_loss: {train_G_epoch_loss:.4f}, D_loss: {train_D_epoch_loss:.4f}, D_acc: {train_D_epoch_acc:.4f} | "
                f"Val - G_loss: {val_G_epoch_loss:.4f}, D_loss: {val_D_epoch_loss:.4f}, D_acc: {val_D_epoch_acc:.4f}"
            )

            #-------------------------------
            # Early Stopping Check
            #-------------------------------
            
            # We use G validation loss for early stopping
            
            # Improvement
            if val_G_epoch_loss < best_val_g_loss:
                best_val_g_loss = val_G_epoch_loss
                best_epoch = epoch
                patience_count = 0
                
                best_G_state = G.state_dict()
                best_D_state = D.state_dict()
                
            else:
                # No improvement
                patience_count += 1

            # If patience is exceeded, stop training
            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                break

        # No validation set up
        else:
            print(
                f"[Epoch {epoch+1}/{max_epochs}] "
                f"Train - D_loss: {train_D_epoch_loss:.4f}, G_loss: {train_G_epoch_loss:.4f}, D_acc: {train_D_epoch_acc:.4f}"
            )

        #----------------------------------------------------------------------
        # Plot images at the end of each epoch
        #----------------------------------------------------------------------
        sample_imgs, sample_masks = next(iter(train_dataloader))
        sample_imgs = sample_imgs.to(device)
        sample_masks = sample_masks.to(device)
        gen_img_plot(G, sample_imgs, sample_masks, epoch)
        
        #----------------------------------------------------------------------
        # Write logs
        #----------------------------------------------------------------------
        log_dict = {
            'train_G_epoch_loss': train_G_epoch_loss,
            'train_D_epoch_loss': train_D_epoch_loss,
            'train_D_epoch_real_acc': train_D_epoch_real_acc,
            'train_D_epoch_fake_acc': train_D_epoch_fake_acc,
            'train_D_epoch_mean_acc': train_D_epoch_acc,
            
            'valid_G_loss': val_G_epoch_loss,
            'valid_D_loss': val_D_epoch_loss,
            'valid_D_acc': val_D_epoch_acc
        }
        
        train_log_file = f"train_log/train_epoch_{epoch}.json"
        with open(train_log_file, mode="w") as f:
            json.dump(log_dict, f, indent=4)

    #==========================================================================
    # Restore the model
    #==========================================================================
    if best_G_state is not None and best_D_state is not None:
        G.load_state_dict(best_G_state)
        D.load_state_dict(best_D_state)
        print(f"Loaded best model from epoch {best_epoch+1} with val G_loss={best_val_g_loss:.4f}.")

    return (G, D, best_epoch)

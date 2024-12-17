# TRAINING LOOP

# initializing lists to track losses for plotting
losses = {
    "G_A": [],           # generator A adversarial loss
    "Cycle_A": [],       # cycle consistency loss A (2D->3D->2D)
    "Cycle_B": [],       # cycle consistency loss B (3D->2D->3D)
    "D": []              # discriminator loss
}

num_epochs = 50
lambda_cycle = 15
adversarial_loss = nn.BCEWithLogitsLoss()
cycle_consistency_loss = nn.L1Loss()
label_smoothing = 0.1

for epoch in range(num_epochs):
    generator_2d_to_3d.train()
    generator_3d_to_2d.train()
    discriminator_3d.train()

    epoch_loss_G_A = 0.0
    epoch_loss_Cycle_A = 0.0
    epoch_loss_Cycle_B = 0.0
    epoch_loss_D = 0.0

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
        frontal = batch['frontal'].to(device)

        # -------------------
        # Train Generator A (2D to 3D)
        # -------------------
        optimizer_G_A.zero_grad()
        fake_3d = generator_2d_to_3d(frontal)
        real_labels = (1.0 - label_smoothing) * torch.ones_like(discriminator_3d(fake_3d)).to(device)
        loss_G_A_adv = adversarial_loss(discriminator_3d(fake_3d), real_labels)

        # Train Generator B (3D to 2D)
        optimizer_G_B.zero_grad()
        reconstructed_xrays = generator_3d_to_2d(fake_3d)
        loss_cycle_A = cycle_consistency_loss(reconstructed_xrays, frontal)

        reconstructed_volumes = generator_2d_to_3d(reconstructed_xrays)
        loss_cycle_B = cycle_consistency_loss(reconstructed_volumes, fake_3d)

        loss_G = loss_G_A_adv + lambda_cycle * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()

        # gradient clipping for the generators
        torch.nn.utils.clip_grad_norm_(generator_2d_to_3d.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(generator_3d_to_2d.parameters(), max_norm=1.0)

        optimizer_G_A.step()
        optimizer_G_B.step()

        # -------------------
        # Train Discriminator (3D)
        # -------------------
        if i % 2 == 0:  # balancing training steps
            optimizer_D.zero_grad()
            real_3d = torch.randn_like(fake_3d).to(device)  # random 3D volume
            real_scores = discriminator_3d(real_3d)
            real_labels = (1.0 - label_smoothing) * torch.ones_like(real_scores).to(device)
            loss_real = adversarial_loss(real_scores, real_labels)

            fake_scores = discriminator_3d(fake_3d.detach())
            fake_labels = label_smoothing * torch.zeros_like(fake_scores).to(device)
            loss_fake = adversarial_loss(fake_scores, fake_labels)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()

            # gradient clipping for the discriminator
            torch.nn.utils.clip_grad_norm_(discriminator_3d.parameters(), max_norm=1.0)

            optimizer_D.step()

        # accumulating losses
        epoch_loss_G_A += loss_G_A_adv.item()
        epoch_loss_Cycle_A += loss_cycle_A.item()
        epoch_loss_Cycle_B += loss_cycle_B.item()
        epoch_loss_D += loss_D.item() if i % 2 == 0 else 0.0

    # calculating average losses
    avg_loss_G_A = epoch_loss_G_A / len(train_loader)
    avg_loss_Cycle_A = epoch_loss_Cycle_A / len(train_loader)
    avg_loss_Cycle_B = epoch_loss_Cycle_B / len(train_loader)
    avg_loss_D = epoch_loss_D / (len(train_loader) / 2)

    # appending epoch losses to the dictionary
    losses["G_A"].append(avg_loss_G_A)
    losses["Cycle_A"].append(avg_loss_Cycle_A)
    losses["Cycle_B"].append(avg_loss_Cycle_B)
    losses["D"].append(avg_loss_D)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] Training: "
        f"Loss_G_A: {avg_loss_G_A:.4f} | "
        f"Loss_Cycle_A: {avg_loss_Cycle_A:.4f} | "
        f"Loss_Cycle_B: {avg_loss_Cycle_B:.4f} | "
        f"Loss_D: {avg_loss_D:.4f}"
    )

    # saving checkpoints every 10 epochs
    if (epoch + 1) % 25 == 0:
        torch.save(generator_2d_to_3d.state_dict(), f"generatorfrontal_2d_to_3d_epoch_{epoch+1}.pth")
        torch.save(generator_3d_to_2d.state_dict(), f"generatorfrontal_3d_to_2d_epoch_{epoch+1}.pth")
        torch.save(discriminator_3d.state_dict(), f"discriminatorfrontal_epoch_{epoch+1}.pth")


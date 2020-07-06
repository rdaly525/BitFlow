def compute_loss(target, y, Range, AreaArray):
    # minimize total area
    target_loss = 1 * torch.sum((AreaArray - 0) ** 2)

    # constraint_1 is the total bits should be greater than 0, will need to pass in range array as well
    constraint_1 = 1000 * torch.sum(torch.max(-(W + Range) + 0.5, torch.zeros(len(W))))

    # constraint_2 y should be within error margin, expected value - calculated value <= 2^-a
    constraint_2 = 1000 * torch.max(torch.abs(Y-y)-2**user_error,torch.zeros(1))

    # target + all constaint losses
    loss = target_loss + constraint_1 + constraint_2
    return loss

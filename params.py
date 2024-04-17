from train import SimpleModel, DeeperModel, WiderModel, NarrowerModel


for model in [SimpleModel, DeeperModel, WiderModel, NarrowerModel]:
    total = 0
    for param in model().parameters():
        total += param.numel()
    print(model, total)

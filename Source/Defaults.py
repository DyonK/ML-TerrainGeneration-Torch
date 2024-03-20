
def HparamDefaults():

    return dict(
        BatchSize = 1,
        LearningRate = 1e-4,
        TrainEpoch=100,
        WeightDecay = 0.0,
        EmaRate = 0.9999,
        DataDropRate = 0.2,
        LogInterval = 100,
        SaveInterval = 1000

    )

def DiffDefaults():
    return dict(
        TimeSteps=1000,
        NoiseSchedule = "Linear",


    )

def ModelDefaults():
    return dict(
        ImageSize = (64,64),
        ConditionalClasses = 31,






    )
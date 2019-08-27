from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Module, LeakyReLU


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.minimal_feature_extractor = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=1),
            ReLU(),


            Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=1),
            ReLU(),

            Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=1),
            LeakyReLU(),
            Conv2d(in_channels=2, out_channels=2, kernel_size=7, stride=1, padding=1),
            LeakyReLU()
        )

    def forward(self, img, model_name='vgg'):
        return self.minimal_feature_extractor(img)


if __name__ == '__main__':
    import torch
    dummy_img = torch.zeros([1, 3, 592, 592])
    conv = Model().forward(dummy_img)
    print(conv.shape)

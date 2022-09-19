from network import deeplab, unet, unext


def get_model(key_number, device):
    model_dic = {0: 'deeplabv3plus_resnet50',
                 1: 'deeplabv3plus_mobilenet',
                 2: 'unet',
                 3: 'unext'
                 }

    if key_number == 0 or key_number == 1:
        net = deeplab.__dict__[model_dic[key_number]](num_classes=1, output_stride=8)

    elif key_number == 2:
        net = unet.UNet(n_channels=3, n_classes=1, bilinear=False)

    elif key_number == 3:
        net = unext.__dict__['UNext'](num_classes=1, input_channel=3, deep_supervision=False)

    else:
        exit("key_number is wrong")

    return net.to(device), model_dic[key_number]

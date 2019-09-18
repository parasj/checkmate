import sys

from integration.tf2.extraction import get_keras_model, MODEL_NAMES

# MODEL_NAMES = ['VGG16', 'VGG19', 'MobileNet', 'fcn_8', 'pspnet', 'vgg_unet', 'unet', 'segnet', 'resnet50_segnet']

if __name__ == "__main__":
    for name in MODEL_NAMES:
        print(name, end=" ")
        try:
            model = get_keras_model(name, input_shape=None)
            print(model.layers[0].input_shape)
        except Exception as e:
            print("ERROR for model", name, e)

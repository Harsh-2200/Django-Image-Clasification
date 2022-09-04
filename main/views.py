
from django.shortcuts import render
from .forms import ImageUpload
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


def handle_file(f):
    with open('img.png', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def home(request):
    context = {}
    return render(request, 'imupload/home.html', context)


def imgprocess(request):
    form = ImageUpload(request.POST, request.FILES)
    if form.is_valid():
        handle_file(request.FILES['image'])

        model = ResNet50(weights='imagenet')
        img_path = 'img.png'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])

        htm = decode_predictions(preds, top=3)[0]
        result = []
        for i in htm:
            result.append((i[1], np.round(i[2]*100, 2)))
        return render(request, 'main/result.html', {'result': result})
    return render(request, 'main/home.html')




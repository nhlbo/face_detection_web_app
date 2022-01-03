from django.shortcuts import render

from darknet import model
from .forms import DetectForm


def index(request):
    if request.method == 'POST':
        form = DetectForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['file']
            choice = form.cleaned_data['model']
            confidence_threshold = form.cleaned_data['threshold']
            if choice == '1':
                img, predicted_time = model.detect_1(img, confidence_threshold)
            else:
                img, predicted_time = model.detect_2(img, confidence_threshold)
            return render(request, 'index.html', {
                'form': form,
                'src': img,
                'predicted_time': predicted_time
            })
    form = DetectForm()
    return render(request, 'index.html', {'form': form})


def about(request):
    return render(request, 'about.html')

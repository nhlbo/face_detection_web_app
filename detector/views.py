from django.shortcuts import render

from darknet import model
from .forms import DetectForm


def index(request):
    if request.method == 'POST':
        form = DetectForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['file']
            confidence_threshold = form.cleaned_data['threshold']
            img, predicted_time = model.detect(img, confidence_threshold)
            return render(request, 'index.html', {
                'form': form,
                'src': img
            })
    form = DetectForm()
    return render(request, 'index.html', {'form': form})

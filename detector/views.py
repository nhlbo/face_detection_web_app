from django.shortcuts import render

from .forms import DetectForm


def index(request):
    if request.method == 'POST':
        form = DetectForm(request.POST, request.FILES)
        if form.is_valid():
            return render(request, 'index.html', {
                'form': form,
                'src': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png'
                       '/640px-Image_created_with_a_mobile_phone.png'
            })
    form = DetectForm()
    return render(request, 'index.html', {'form': form})

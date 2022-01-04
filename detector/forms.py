from django import forms


class DetectForm(forms.Form):
    threshold = forms.FloatField(label='threshold')
    model = forms.CharField(label='model')
    file = forms.FileField(widget=forms.FileInput(
        attrs={
            'id': 'formFile',
            'class': 'form-control',
            'type': 'file',
            'name': 'file',
            'accept': '.jpg,.jpeg,.png',
            'onchange': 'preview()'
        }
    ))

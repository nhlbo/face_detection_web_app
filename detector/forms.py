from django import forms


class DetectForm(forms.Form):
    threshold = forms.FloatField(label='threshold')
    model = forms.CharField(label='model')
    file = forms.FileField()

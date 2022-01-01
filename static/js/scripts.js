let slider = document.getElementById("customRange1");
let threshold = document.getElementById("threshold");

threshold.innerHTML = Number(slider.value).toFixed(2);

slider.oninput = function() {
  threshold.innerHTML = Number(this.value).toFixed(2);
}
window.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById("drawingCanvas");
  const canvasContainer = document.getElementById("canvasContainer");
  const ctx = canvas.getContext("2d");
  const brushSize = document.getElementById("brushSize");
  const brushColor = document.getElementById("brushColor");
  const clearBtn = document.getElementById("clearButton");
  const saveBtn = document.getElementById("saveButton");
  const fullscreenBtn = document.getElementById("fullscreenButton");
  const saveForm = document.getElementById("saveForm");
  const imageInput = document.getElementById("imageData");
  const modelSelect = document.getElementById("modelSelect");
  const showBtn = document.getElementById("showClassesBtn");
  const classList = document.getElementById("classList");

  let isFullscreen = false;

  // Tam ekran fonksiyonu
  fullscreenBtn.addEventListener("click", () => {
    if (!isFullscreen) {
      if (canvasContainer.requestFullscreen) {
        canvasContainer.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  });

  // Tam ekran değişikliğini izle
  document.addEventListener('fullscreenchange', () => {
    isFullscreen = !!document.fullscreenElement;
    fullscreenBtn.textContent = isFullscreen ? "Küçült" : "Tam Ekran";

    if (isFullscreen) {
      canvas.style.width = '90vw';
      canvas.style.height = '80vh';
    } else {
      canvas.style.width = '600px';
      canvas.style.height = '400px';
    }
  });

  // Beyaz zemin
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let painting = false;
  function startPosition(e) {
    painting = true;
    draw(e);
  }
  function endPosition() {
    painting = false;
    ctx.beginPath();
  }
  function draw(e) {
    if (!painting) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    // Canvas boyutuna göre koordinatları ölçekle
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    ctx.lineWidth = brushSize.value;
    ctx.lineCap = "round";
    ctx.strokeStyle = brushColor.value;
    ctx.lineTo(x * scaleX, y * scaleY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x * scaleX, y * scaleY);
    e.preventDefault();
  }

  // Event listener'lar
  canvas.addEventListener("mousedown", startPosition);
  canvas.addEventListener("touchstart", startPosition, { passive: false });
  canvas.addEventListener("mouseup", endPosition);
  canvas.addEventListener("touchend", endPosition, { passive: false });
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("touchmove", draw, { passive: false });

  // Diğer fonksiyonlar aynı kalacak
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
  });

  saveBtn.addEventListener("click", (e) => {
    e.preventDefault();
    const temp = document.createElement("canvas");
    const tctx = temp.getContext("2d");
    temp.width = 256;
    temp.height = 256;
    tctx.fillStyle = "#FFFFFF";
    tctx.fillRect(0, 0, 256, 256);
    tctx.drawImage(canvas, 0, 0, 256, 256);
    imageInput.value = temp.toDataURL("image/png");
    saveForm.submit();
  });

  showBtn.addEventListener("click", async () => {
    const model = modelSelect.value;
    classList.innerHTML = 'Yükleniyor…';
    try {
      const res = await fetch(`/get_classes/?model_name=${encodeURIComponent(model)}`);
      if (!res.ok) throw new Error(`Sunucu hatası (${res.status})`);
      const data = await res.json();
      classList.innerHTML = `
        <h4>${model} sınıfları (${data.classes.length} adet):</h4>
        <ul>${data.classes.map(c => `<li>${c}</li>`).join('')}</ul>
      `;
    } catch (err) {
      classList.innerHTML = `<span style="color:red">Hata: ${err.message}</span>`;
    }
  });

  modelSelect.addEventListener("change", () => {
    classList.innerHTML = '';
  });
});
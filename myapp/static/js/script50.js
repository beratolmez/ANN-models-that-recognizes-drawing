// script50.js - home50.html için özel çizim scripti
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");

  // Ekranda beyaz zemin + siyah çizim
  const BG_COLOR   = '#FFFFFF';
  const DRAW_COLOR = '#000000';

  // Çizim ayarları
  let isDrawing = false;
  let lastX = 0, lastY = 0;

  // UI elementleri
  const brushSize  = document.getElementById("brushSize");
  const brushColor = document.getElementById("brushColor");
  const clearBtn   = document.getElementById("clearButton");
  const saveBtn    = document.getElementById("saveButton");

  // Canvas’ı beyaz yap, çizgi ayarlarını siyah olarak sabitle
  function initCanvas() {
    ctx.fillStyle   = BG_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = DRAW_COLOR;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    brushColor.value = DRAW_COLOR;
  }

  function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX  || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY  || (e.touches && e.touches[0].clientY);
    return [clientX - rect.left, clientY - rect.top];
  }

  function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getPosition(e);
  }
  function stopDrawing() {
    isDrawing = false;
  }
  function draw(e) {
    if (!isDrawing) return;
    const [x, y] = getPosition(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.lineWidth   = brushSize.value;
    ctx.strokeStyle = brushColor.value;
    ctx.stroke();
    [lastX, lastY] = [x, y];
    e.preventDefault();
  }

  function clearCanvas() {
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function saveDrawing() {
    // Model input için 64×64 geçici canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width  = 64;
    tempCanvas.height = 64;
    const tempCtx = tempCanvas.getContext('2d');

    // Öncelikle beyaz zemin (canvas’ı tamamen örtmesin diye isteğe bağlı)
    tempCtx.fillStyle = BG_COLOR;
    tempCtx.fillRect(0, 0, 64, 64);

    // Orijinal çizimi 64×64 olarak kopyala
    tempCtx.drawImage(canvas, 0, 0, 64, 64);

    // Form’a base64 ata ve gönder
    const imageData = tempCanvas.toDataURL('image/png');
    document.getElementById("imageData").value = imageData;
    document.getElementById("saveForm").submit();
  }

  // Olay dinleyicileri
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup',   stopDrawing);
  canvas.addEventListener('mouseout',  stopDrawing);

  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e.touches[0]); });
  canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); draw(e.touches[0]); });
  canvas.addEventListener('touchend',   stopDrawing);

  clearBtn.addEventListener('click', clearCanvas);
  saveBtn.addEventListener('click', saveDrawing);

  // Başlat
  initCanvas();
});

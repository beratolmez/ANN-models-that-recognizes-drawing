import uuid
import base64
import io

import numpy as np
from django.shortcuts import render
from django.core.files.base import ContentFile
from PIL import Image
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from .ml50 import *
from .models import Drawing
from .ml import predict_file, available_models, models_meta
from .mlgenerate import SketchGenCond, sample_sequence, CLASSES as GEN_CLASSES


DEVICE = torch.device("cpu")


APP_DIR = os.path.dirname(__file__)
GEN_MODEL_PATH = os.path.join(APP_DIR, "ml_models", "resim_cizen.pth")

gen_model = SketchGenCond().to(DEVICE)
if not os.path.exists(GEN_MODEL_PATH):
    raise FileNotFoundError(f"Generation model not found: {GEN_MODEL_PATH!r}")
gen_model.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE))
gen_model.eval()
def home(request):
    saved_image_url     = None
    processed_image_data = None
    prediction          = None
    model_list          = available_models

    if request.method == "POST":

        fmt, imgstr = request.POST["image"].split(";base64,")
        img = Image.open(io.BytesIO(base64.b64decode(imgstr))).convert("RGBA")


        buf1 = io.BytesIO()
        img.resize((280, 280), Image.Resampling.LANCZOS).save(buf1, format="PNG")
        buf1.seek(0)
        drawing = Drawing.objects.create(
            image=ContentFile(buf1.read(), name=f"drawing_{uuid.uuid4()}.png")
        )
        saved_image_url = drawing.image.url


        meta = request.POST.get("model_name", model_list[0])
        proc = models_meta[meta]['preprocess']
        x = proc(drawing.image.path)


        if isinstance(x, torch.Tensor):
            arr = x.squeeze().cpu().numpy()
        else:
            arr = x.squeeze()


        if arr.ndim == 1:
            side = int(np.sqrt(arr.shape[0]))
            arr2 = arr.reshape(side, side)
        elif arr.ndim == 3:

            arr2 = arr.reshape(arr.shape[-2], arr.shape[-1])
        else:

            arr2 = arr


        pil28 = Image.fromarray((arr2 * 255).astype(np.uint8), mode="L")
        buf2 = io.BytesIO()
        pil28.save(buf2, format="PNG")
        processed_image_data = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode()


        if 'predict' in models_meta[meta]:
            prediction = models_meta[meta]['predict'](drawing.image.path)
        else:
            prediction = predict_file(meta, drawing.image.path)

    return render(request, "home.html", {
        "model_list": model_list,
        "saved_image_url": saved_image_url,
        "processed_image_data": processed_image_data,
        "prediction": prediction,
    })





def home50(request):
    saved_image_url = None
    prediction50 = None
    processed64 = None
    probabilities50 = None

    if request.method == "POST":
        fmt, imgstr = request.POST["image"].split(";base64,")
        img_bytes = base64.b64decode(imgstr)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        buf1 = io.BytesIO()
        img.resize((280, 280), Image.Resampling.LANCZOS).save(buf1, format="PNG")
        buf1.seek(0)
        name = f"drawing_{uuid.uuid4()}.png"
        drawing = Drawing.objects.create(
            image=ContentFile(buf1.read(), name=name)
        )
        saved_image_url = drawing.image.url

        pil64 = Image.open(drawing.image.path).convert("L").resize((64, 64), Image.Resampling.NEAREST)
        pil64 = ImageOps.invert(pil64)

        buf2 = io.BytesIO()
        pil64.save(buf2, format="PNG")
        processed64 = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode()

        prediction50, probs50 = predict_file50(drawing.image.path)
        probabilities50 = sorted(zip(classes50, probs50.tolist()), key=lambda x: x[1], reverse=True)

    return render(request, "home50.html", {
        "saved_image_url": saved_image_url,
        "processed64": processed64,
        "prediction50": prediction50,
        "probabilities50": probabilities50,
    })


def generate(request):
    generated_image = None

    if request.method == "POST":
        cls_id = int(request.POST.get("class_id", 0))
        seq = sample_sequence(gen_model, cls_id)

        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        x = y = 0
        for dx, dy, p1, p2, p3 in seq:
            nx, ny = x + dx, y + dy
            if p1:
                ax.plot([x, nx], [y, ny], linewidth=2)
            x, y = nx, ny

        ax.invert_yaxis()
        ax.axis("off")

        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        generated_image = f"data:image/png;base64,{b64}"

    return render(request, "generate.html", {
        "classes": GEN_CLASSES,
        "generated_image": generated_image,
    })



from django.http import JsonResponse

def get_model_classes(request):
    model_name = request.GET.get('model_name')
    meta = models_meta.get(model_name)
    if not meta:
        return JsonResponse({'error': 'Model bulunamadı.'}, status=404)
    return JsonResponse({'classes': meta['classes']})

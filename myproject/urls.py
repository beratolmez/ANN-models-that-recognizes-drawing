"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static

from myapp.views import generate, home50, home, get_model_classes
from myproject import settings

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', home, name='home'),
 path('predict50/', home50, name='predict50'),
path("generate/", generate, name="generate"),
path('get_classes/', get_model_classes, name='get_model_classes'),
]

# Medya dosyalarını geliştirme ortamında sunmak için
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

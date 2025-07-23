from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.utils.html import format_html
from .models import Drawing


class DrawingAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'image_tag')  # Admin panelinde gösterilecek sütunlar

    def image_tag(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" style="border-radius:5px;"/>', obj.image.url)
        return "(Resim Yok)"

    image_tag.short_description = 'Önizleme'  # Admin panelindeki sütun adı


admin.site.register(Drawing, DrawingAdmin)

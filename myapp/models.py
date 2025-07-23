# Create your models here.
from django.db import models
import uuid

class Drawing(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='drawings/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Drawing {self.id} - {self.created_at}"

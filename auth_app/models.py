import uuid

from django.db import models
from django.contrib.auth.models import AbstractUser, PermissionsMixin, BaseUserManager


class UserManager(BaseUserManager):
    def create_user(self,email,password=None,**extra_fields):

        if not email:
            raise ValueError("Email is required")

        email=self.normalize_email(email)

        user=self.model(email=email,**extra_fields)

        user.set_password(password)

        user.save(using=self._db)

        return user


    def create_superuser(self,email,password=None,**extra_fields):
        extra_fields.setdefault("is_staff",True)
        extra_fields.setdefault("is_superuser",True)
        extra_fields.setdefault("is_active",True)

        return self.create_user(email,password,**extra_fields)


class User(AbstractUser):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    username=None
    email=models.EmailField(unique=True)
    phone_number=models.CharField(max_length=15,blank=True,null=True)
    objects=UserManager()

    USERNAME_FIELD="email"

    REQUIRED_FIELDS=[]

    def __str__(self):
        return self.email

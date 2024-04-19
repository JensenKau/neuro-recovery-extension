from __future__ import annotations

from django.urls import path, include

from .views import *


user_urls = [
    path("create_user/", CreateUserView.as_view()),
]

main_urls = [
    path("user/", include(user_urls))
]
 
if __name__ == "__main__":
    pass
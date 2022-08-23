"""mn URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls import url
from APP import zdd_view
from APP import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/',views.upload),
    path('down/',views.down),
    path('deldownfile/',views.deldownfile),
    path('yuanshi/',views.yuanshi),
    path('xiangdui/',views.xiangdui),
    path('nfyuanshi/',views.nfYuanshi),
    path('nfxiangdui/',views.nfXiangdui),
    path('tosql/',views.tosql),
    path('login/',views.login),
    path('volin/',views.volin),
    path('register/', views.register),
    url(r'^fanshu', views.fanshu),
    url(r'^searchdis', zdd_view.searchdis),
    url(r'^disnew', zdd_view.disnew),
    url(r'^charts', zdd_view.charts),
    url(r'^PCA_MDS', zdd_view.PCA_MDS),
    url(r'^oplds', zdd_view.oplds),
    url(r'^mama', zdd_view.mama),
]

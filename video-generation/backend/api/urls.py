from django.urls import path
from .views import TaskStatusView, TestTaskView
from .views import GenerateAndUploadShortView

urlpatterns = [
    path("test-task/", TestTaskView.as_view()),
    path("task-status/<str:task_id>/", TaskStatusView.as_view()),
    path("generate-and-upload-short/", GenerateAndUploadShortView.as_view()),
]

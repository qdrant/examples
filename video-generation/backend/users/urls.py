from django.urls import path
from .views import RegisterView, LoginView
from .views import SaveYouTubeTokenView
from .views import VideoListCreateView, VideoRetrieveUpdateDestroyView
from .views import UpdateAPIKeysView
from .views import UserAPIKeysView
from .views import YouTubeOAuthCallbackView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path("youtube/token/", SaveYouTubeTokenView.as_view(), name="save-youtube-token"),
    path("youtube/callback/", YouTubeOAuthCallbackView.as_view(), name="youtube-oauth-callback"),

    path("videos/", VideoListCreateView.as_view(), name="video-list-create"),
    path('videos/<uuid:pk>/', VideoRetrieveUpdateDestroyView.as_view(), name='video-detail'),
    path('update-keys/', UpdateAPIKeysView.as_view(), name='update-api-keys'),
    path("api-keys/", UserAPIKeysView.as_view(), name="get-api-keys"),

]


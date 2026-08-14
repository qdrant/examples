from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import RegisterSerializer, LoginSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from api.youtube_utils import get_authenticated_channel_id 
from rest_framework import generics, permissions
from .models import Video
from .serializers import VideoSerializer
from .models import YouTubeToken
from .serializers import YouTubeTokenSerializer
from rest_framework.permissions import IsAuthenticated
from api.tasks import process_youtube_channel
from .serializers import UpdateAPIKeysSerializer
from .serializers import UserAPIKeysSerializer


import requests
from django.utils.dateparse import parse_datetime
from django.utils import timezone
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI") 

class YouTubeOAuthCallbackView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        code = request.data.get("code")
        if not code:
            return Response({"error": "Missing code"}, status=400)

        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": request.data.get("client_id"),
            "client_secret": request.data.get("client_secret"),
            "redirect_uri": request.data.get("redirect_uri"),
            "grant_type": "authorization_code"
        }

        token_response = requests.post(token_url, data=data)
        if not token_response.ok:
            return Response({"error": "Failed to fetch tokens", "details": token_response.json()}, status=400)

        token_data = token_response.json()

        # Store in DB
        token_obj, _ = YouTubeToken.objects.update_or_create(
            user=request.user,
            defaults={
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": data["client_id"],
                "client_secret": data["client_secret"],
                "scopes": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly",
                "expiry": datetime.utcnow() + timedelta(seconds=token_data["expires_in"])
            }
        )

        return Response({"message": "Token saved successfully"})

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            refresh = RefreshToken.for_user(user)
            return Response({
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            })
        return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)

class SaveYouTubeTokenView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        code = request.data.get('code')
        if not code:
            return Response({"detail": "Missing authorization code."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Exchange code for tokens
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            }
            token_response = requests.post(token_url, data=data)
            token_response.raise_for_status()
            token_data = token_response.json()

            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")
            token_uri = token_url
            scopes = token_data.get("scope")
            expires_in = token_data.get("expires_in")  # seconds

            expiry = timezone.now() + timezone.timedelta(seconds=expires_in)

            # Save to database
            youtube_token, created = YouTubeToken.objects.update_or_create(
                user=request.user,
                defaults={
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_uri": token_uri,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "scopes": scopes,
                    "expiry": expiry,
                }
            )

            # Fetch channel ID and update
            try:
                channel_id = get_authenticated_channel_id(youtube_token)
                youtube_token.channel_id = channel_id
                youtube_token.save()

                process_youtube_channel.delay(channel_id, str(request.user.id))
            except Exception as e:
                print(f"⚠️ Failed to fetch channel ID: {e}")

            return Response({"detail": "YouTube token saved successfully."}, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to exchange code: {e}")
            return Response({"detail": "Failed to exchange code for tokens."}, status=status.HTTP_400_BAD_REQUEST)

class VideoRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)

class VideoListCreateView(generics.ListCreateAPIView):
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class UpdateAPIKeysView(generics.UpdateAPIView):
    serializer_class = UpdateAPIKeysSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

class UserAPIKeysView(generics.RetrieveAPIView):
    serializer_class = UserAPIKeysSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user
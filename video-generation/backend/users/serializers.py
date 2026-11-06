from rest_framework import serializers
from .models import User
from django.contrib.auth import authenticate
from .models import YouTubeToken
from .models import Video
from rest_framework import generics, permissions


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['id', 'title', 'description', 'video_url', 'created_at']
        read_only_fields = ['id', 'created_at']


class VideoListCreateView(generics.ListCreateAPIView):
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Invalid credentials")

class YouTubeTokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = YouTubeToken
        fields = "__all__"
        read_only_fields = ['user']

class UpdateAPIKeysSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'openai_api_key',
            'replicate_api_key',
            'elevenlabs_api_key',
            'openai_model',
            'elevenlabs_voice_id',
            'audience',
            'flux_model',
        ]
        extra_kwargs = {
            'openai_api_key': {'write_only': True},
            'replicate_api_key': {'write_only': True},
            'elevenlabs_api_key': {'write_only': True},
        }


class UserAPIKeysSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "openai_api_key",
            "replicate_api_key",
            "elevenlabs_api_key",
            "openai_model",
            "elevenlabs_voice_id",
            "audience",
            "flux_model",
        ]
        extra_kwargs = {
            "openai_api_key": {"read_only": True},
            "replicate_api_key": {"read_only": True},
            "elevenlabs_api_key": {"read_only": True},
        }
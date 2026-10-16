from django.contrib import admin
from .models import User, YouTubeToken, Video

admin.site.register(User)
admin.site.register(Video)

class YouTubeTokenAdmin(admin.ModelAdmin):
    list_display = ['user', 'channel_id', 'expiry']

admin.site.register(YouTubeToken, YouTubeTokenAdmin)

import os
from django.http import JsonResponse
from dotenv import load_dotenv

load_dotenv()

class APIKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.api_key = os.getenv("X-API-KEY")
        self.exempt_paths = [
            "/api/users/register/",
            "/api/users/login/",
            "/api/users/",
            "/admin/",
            "/admin/login/",
            "/admin/logout/",
            "/favicon.ico",
        ]

    def __call__(self, request):
        # Skip API key check for exempt paths
        if any(request.path.startswith(path) for path in self.exempt_paths):
            return self.get_response(request)

        # Check API key
        key = request.headers.get("X-API-KEY")
        if not key or key != self.api_key:
            return JsonResponse({"detail": "Unauthorized: Invalid or missing API Key."}, status=401)

        return self.get_response(request)

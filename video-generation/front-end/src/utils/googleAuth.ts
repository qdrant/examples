// src/utils/googleAuth.ts
export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID
export const REDIRECT_URI = import.meta.env.VITE_REDIRECT_URI; // Must match your Google console settings
export const SCOPES = [
  "https://www.googleapis.com/auth/youtube.upload",
  "https://www.googleapis.com/auth/youtube.readonly"
];
// src/utils/googleAuth.ts
export function buildGoogleAuthUrl(): string {
    const baseUrl = "https://accounts.google.com/o/oauth2/v2/auth";
    const params = new URLSearchParams({
      client_id: GOOGLE_CLIENT_ID,
      redirect_uri: REDIRECT_URI,
      response_type: "code",
      access_type: "offline",      // needed to get refresh_token
      prompt: "consent",           // force refresh scope prompt
      scope: SCOPES.join(" ")
    });
  
    return `${baseUrl}?${params.toString()}`;
  }
  
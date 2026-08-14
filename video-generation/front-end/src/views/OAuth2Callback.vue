

<script lang="ts" setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

onMounted(async () => {
  const params = new URLSearchParams(window.location.search)
  const code = params.get('code')

  if (!code) {
    console.error("No code found in URL")
    return
  }

  try {
    // 1. Get your app's access token from localStorage
    const appAccessToken = localStorage.getItem('accessToken')

    if (!appAccessToken) {
      throw new Error('App access token missing in localStorage')
    }

    // 2. Send only the code to your backend
    await axios.post('/api/users/youtube/token/', {
      code: code
    }, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${appAccessToken}`,
        'X-API-KEY': import.meta.env.VITE_X_API_KEY
      }
    })

    console.log("✅ YouTube token saved successfully")

    // 3. Redirect to dashboard or wherever you want
    router.push('/create')
  } catch (error) {
    console.error("❌ Failed to handle OAuth callback", error)
    alert("Something went wrong. Check console logs.")
  }
})
</script>

<template>
  <div class="flex flex-col items-center justify-center h-screen">
    <p class="text-lg font-semibold">Finishing up authentication with YouTube...</p>
    <p class="text-sm text-gray-500 mt-2">Please wait, redirecting...</p>
  </div>
</template>

<style scoped>
</style>

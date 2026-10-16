<template>
  <div class="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4">
    <h1 class="text-4xl font-bold text-white mb-8">Connect Your YouTube Channel</h1>
    <p class="text-lg text-gray-200 max-w-2xl text-center mb-8">
      Connect your YouTube channel to start generating AI-powered Shorts automatically. Vidyne will help you create engaging content consistently. 
      Ignore if already connected.
    </p>
    <button 
      @click="connectYouTube"
      class="bg-red-600 hover:bg-red-700 text-white text-lg font-semibold px-8 py-4 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200"
    >
      Connect YouTube Channel
    </button>
  </div>
</template>

<script lang="ts" setup>
import { buildGoogleAuthUrl } from "../utils/googleAuth";

function connectYouTube() {
  const url = buildGoogleAuthUrl();
  window.location.href = url;
}
</script>

<style scoped>
/* Any existing styles can be removed as we're using Tailwind classes */
</style>

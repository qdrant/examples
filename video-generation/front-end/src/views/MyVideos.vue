<template>
  <div class="w-full max-w-full mx-auto mt-10 p-6 bg-gray-900 rounded-xl shadow flex flex-col gap-8 h-full min-h-screen">
    <h1 class="text-2xl font-bold mb-4 text-white">My Videos</h1>
    <div v-if="loading" class="text-white">Loading...</div>
    <div v-else-if="videos.length === 0" class="text-white">No videos found.</div>
    <div v-else>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div v-for="video in paginatedVideos" :key="video.id" class="bg-gray-800 rounded-lg p-4 flex flex-col shadow-lg border border-gray-700">
          <div class="aspect-video w-full mb-4">
            <iframe 
              :src="getEmbedUrl(video.video_url)" 
              class="w-full h-full rounded"
              frameborder="0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowfullscreen
            ></iframe>
          </div>
          <h3 class="font-semibold text-sm mb-2 line-clamp-2 text-white">{{ video.title }}</h3>
          <p class="text-xs text-gray-300 mb-4 line-clamp-3">{{ video.description }}</p>
          <div class="mt-auto flex items-center justify-between">
            <a 
              :href="video.video_url" 
              target="_blank"
              class="text-blue-400 hover:text-blue-300 text-sm font-medium"
            >
              Watch on YouTube
            </a>
            <div class="text-xs text-gray-300">{{ formatDate(video.created_at) }}</div>
          </div>
        </div>
      </div>
      <div v-if="totalPages > 1" class="flex justify-center mt-6 gap-2">
        <button
          class="px-4 py-2 rounded bg-blue-600 text-white font-semibold disabled:opacity-50 disabled:bg-blue-400 hover:bg-blue-700"
          :disabled="currentPage === 1"
          @click="currentPage--"
        >
          Previous
        </button>
        <span class="px-4 py-2 text-white">Page {{ currentPage }} of {{ totalPages }}</span>
        <button
          class="px-4 py-2 rounded bg-blue-600 text-white font-semibold disabled:opacity-50 disabled:bg-blue-400 hover:bg-blue-700"
          :disabled="currentPage === totalPages"
          @click="currentPage++"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';

interface Video {
  id: string;
  title: string;
  description: string;
  video_url: string;
  created_at: string;
}

const VIDEOS_PER_PAGE = 3;
const currentPage = ref(1);
const videos = ref<Video[]>([]);
const loading = ref(true);
const router = useRouter();

const paginatedVideos = computed(() => {
  const start = (currentPage.value - 1) * VIDEOS_PER_PAGE;
  return videos.value.slice(start, start + VIDEOS_PER_PAGE);
});

const totalPages = computed(() => Math.ceil(videos.value.length / VIDEOS_PER_PAGE));

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function getEmbedUrl(url: string) {
  const videoId = url.split('watch?v=')[1];
  return `https://www.youtube.com/embed/${videoId}`;
}

onMounted(async () => {
  const accessToken = localStorage.getItem('accessToken');
  if (!accessToken) {
    router.push('/signin');
    return;
  }

  try {
    const res = await fetch('/api/users/videos/', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
        'X-API-KEY': import.meta.env.VITE_X_API_KEY
      },
    });
    
    if (!res.ok) throw new Error('Failed to fetch videos');
    videos.value = await res.json();
  } catch (e) {
    console.error('Error fetching videos:', e);
    videos.value = [];
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Prevent horizontal scrolling */
.body, .html {
  overflow-x: hidden !important;
}

/* Ensure main container never goes wider than viewport */
.w-full {
  width: 100% !important;
}
.max-w-full {
  max-width: 100vw !important;
}
</style>

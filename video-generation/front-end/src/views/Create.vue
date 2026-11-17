<template>
  <div class="min-h-screen w-full bg-gray-50 text-gray-900 flex flex-col">
    <div class="max-w-3xl mx-auto w-full mt-10 p-6 bg-white rounded-xl shadow flex flex-col gap-8">
      <h1 class="text-2xl font-bold mb-2">Automated Video Generation</h1>
      
      <div class="prose prose-sm">
        <p class="text-gray-700 leading-relaxed">
          Vidyne automatically gnerates and posts new shorts 3 times a day based on your existing videos. 
          We use the Qdrant to create a vector embedding of your existing content. 
          Then based on the audience information provided in the settings panel, the system will come up with three shorts ideas everyday and upload them to your YouTube. 
          Ensure that you have connected your YouTube channel on the home page. 
          For best results, enter a comprehensive audience text on the settings panel. 
          This application works best for channels with atleast 5 - 10 videos that the system can learn from. 
          If you are just starting, we recommend that you first upload a few shorts before trying Vidyne. 
          At the moment, Vidyne generates shorts only.
        </p>
        <p class="mt-4 font-medium">Click the button below to generate a new video now.</p>
      </div>

      <div class="flex flex-col items-center justify-center py-8">
        <button 
          @click="generateVideos"
          :disabled="loading"
          class="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-lg font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ loading ? 'Generating...' : 'Generate Video' }}
        </button>
      </div>

      <div v-if="taskId" class="mt-4 p-6 bg-blue-50 rounded-lg">
        <h3 class="text-lg font-semibold text-blue-900 mb-2">Video Generation Started!</h3>
        <p class="text-blue-800">Task ID: {{ taskId }}</p>
        <p class="mt-4 text-blue-700">
          Time to grab a coffee! We'll send you an email when your video is ready and uploaded to YouTube.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const loading = ref(false);
const taskId = ref<string | null>(null);

const generateVideos = async () => {
  loading.value = true;
  const accessToken = localStorage.getItem('accessToken');

  try {
    const res = await fetch('/api/generate-and-upload-short/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
        'X-API-KEY': import.meta.env.VITE_X_API_KEY,
      }
    });

    if (!res.ok) {
      throw new Error(`Error: ${res.status}`);
    }

    const data = await res.json();
    if (data.task_id) {
      taskId.value = data.task_id;
    }
  } catch (error) {
    console.error('Failed to generate video:', error);
    // You might want to add error handling UI here
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.prose {
  max-width: none;
}
</style>

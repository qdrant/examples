<template>
  <div class="max-w-3xl mx-auto mt-10 p-6 bg-gray-800 rounded-xl shadow text-white">
    <h1 class="text-2xl font-bold mb-6 text-white">API Settings</h1>
    
    <form @submit.prevent="saveSettings" class="space-y-6">
      <!-- OpenAI -->
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">OpenAI API Key</label>
        <input 
          v-model="settings.openai_api_key" 
          type="password" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
          placeholder="sk-..."
        />
      </div>

      <!-- OpenAI Model -->
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">OpenAI Model</label>
        <select 
          v-model="settings.openai_model" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
        >
          <option value="">Select a model</option>
          <optgroup label="GPT-4.1">
            <option value="gpt-4.1">gpt-4.1 ($2.00/$0.50/$8.00)</option>
            <option value="gpt-4.1-2025-04-14">gpt-4.1-2025-04-14 ($2.00/$0.50/$8.00)</option>
          </optgroup>
          <optgroup label="GPT-4.1 Mini">
            <option value="gpt-4.1-mini">gpt-4.1-mini ($0.40/$0.10/$1.60)</option>
            <option value="gpt-4.1-mini-2025-04-14">gpt-4.1-mini-2025-04-14 ($0.40/$0.10/$1.60)</option>
          </optgroup>
          <optgroup label="GPT-4.1 Nano">
            <option value="gpt-4.1-nano">gpt-4.1-nano ($0.10/$0.025/$0.40)</option>
            <option value="gpt-4.1-nano-2025-04-14">gpt-4.1-nano-2025-04-14 ($0.10/$0.025/$0.40)</option>
          </optgroup>
          <optgroup label="GPT-4.5">
            <option value="gpt-4.5-preview">gpt-4.5-preview ($75.00/$37.50/$150.00)</option>
            <option value="gpt-4.5-preview-2025-02-27">gpt-4.5-preview-2025-02-27 ($75.00/$37.50/$150.00)</option>
          </optgroup>
          <optgroup label="GPT-4O">
            <option value="gpt-4o">gpt-4o ($2.50/$1.25/$10.00)</option>
            <option value="gpt-4o-2024-08-06">gpt-4o-2024-08-06 ($2.50/$1.25/$10.00)</option>
            <option value="gpt-4o-audio-preview">gpt-4o-audio-preview ($2.50/-/$10.00)</option>
            <option value="gpt-4o-audio-preview-2024-12-17">gpt-4o-audio-preview-2024-12-17 ($2.50/-/$10.00)</option>
            <option value="gpt-4o-realtime-preview">gpt-4o-realtime-preview ($5.00/$2.50/$20.00)</option>
            <option value="gpt-4o-realtime-preview-2024-12-17">gpt-4o-realtime-preview-2024-12-17 ($5.00/$2.50/$20.00)</option>
          </optgroup>
          <optgroup label="GPT-4O Mini">
            <option value="gpt-4o-mini">gpt-4o-mini ($0.15/$0.075/$0.60)</option>
            <option value="gpt-4o-mini-2024-07-18">gpt-4o-mini-2024-07-18 ($0.15/$0.075/$0.60)</option>
            <option value="gpt-4o-mini-audio-preview">gpt-4o-mini-audio-preview ($0.15/-/$0.60)</option>
            <option value="gpt-4o-mini-audio-preview-2024-12-17">gpt-4o-mini-audio-preview-2024-12-17 ($0.15/-/$0.60)</option>
            <option value="gpt-4o-mini-realtime-preview">gpt-4o-mini-realtime-preview ($0.60/$0.30/$2.40)</option>
            <option value="gpt-4o-mini-realtime-preview-2024-12-17">gpt-4o-mini-realtime-preview-2024-12-17 ($0.60/$0.30/$2.40)</option>
          </optgroup>
          <optgroup label="O Series">
            <option value="o1">o1 ($15.00/$7.50/$60.00)</option>
            <option value="o1-2024-12-17">o1-2024-12-17 ($15.00/$7.50/$60.00)</option>
            <option value="o1-pro">o1-pro ($150.00/-/$600.00)</option>
            <option value="o1-pro-2025-03-19">o1-pro-2025-03-19 ($150.00/-/$600.00)</option>
            <option value="o3">o3 ($10.00/$2.50/$40.00)</option>
            <option value="o3-2025-04-16">o3-2025-04-16 ($10.00/$2.50/$40.00)</option>
            <option value="o4-mini">o4-mini ($1.10/$0.275/$4.40)</option>
            <option value="o4-mini-2025-04-16">o4-mini-2025-04-16 ($1.10/$0.275/$4.40)</option>
            <option value="o3-mini">o3-mini ($1.10/$0.55/$4.40)</option>
            <option value="o3-mini-2025-01-31">o3-mini-2025-01-31 ($1.10/$0.55/$4.40)</option>
            <option value="o1-mini">o1-mini</option>
            <option value="o1-mini-2024-09-12">o1-mini-2024-09-12</option>
          </optgroup>
        </select>
      </div>
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">Image Generation Model</label>
        <select 
          v-model="settings.flux_model" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
        >
          <option value="">Select image model</option>
            <option value="black-forest-labs/flux-schnell"> black-forest-labs/flux-schnell $0.003 / image lower quality best for testing</option>
            <option value="black-forest-labs/flux-1.1-pro">black-forest-labs/flux-1.1-pro $0.04 / image Higher quality, best for production
              $0.04 / image</option>
         
        </select>
      </div>
      <!-- Replicate -->
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">Replicate API Key</label>
        <input 
          v-model="settings.replicate_api_key" 
          type="password" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
          placeholder="r8_..."
        />
      </div>

      <!-- ElevenLabs -->
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">ElevenLabs API Key</label>
        <input 
          v-model="settings.elevenlabs_api_key" 
          type="password" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
          placeholder="sk_..."
          @input="handleElevenLabsKeyChange"
        />
      </div>

      <!-- ElevenLabs Voice -->
      <div v-if="voices.length > 0">
        <label class="block text-sm font-medium text-gray-200 mb-1">ElevenLabs Voice</label>
        <select 
          v-model="settings.elevenlabs_voice_id"
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
        >
          <option v-for="voice in voices" :key="voice.voice_id" :value="voice.voice_id">
            {{ voice.name }} ({{ voice.labels?.gender }} - {{ voice.labels?.accent }})
          </option>
        </select>
      </div>
      
      <!-- Voice ID Manual Input -->
      <div v-else>
        <label class="block text-sm font-medium text-gray-200 mb-1">ElevenLabs Voice ID</label>
        <input 
          v-model="settings.elevenlabs_voice_id" 
          type="text" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
          placeholder="Voice ID..."
        />
      </div>

      <!-- Audience -->
      <div>
        <label class="block text-sm font-medium text-gray-200 mb-1">Target Audience</label>
        <textarea 
          v-model="settings.audience" 
          class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white h-32 resize-y"
          placeholder="Tech lovers and content creators interested in Large Language Models, Generative AI, model context protocol, AI MCPs, Vibe Coding, AI agents, AI tools, AI productivity, AI video generation, AI video editing, AI content generation, AI image generation, AI voice generation, AI video creation, AI automation, AI for creators, AI for productivity, AI for solopreneurs, AI for entrepreneurs, AI for startups, AI for business, AI for tech, AI for marketing, AI for sales, AI for content creators, AI for YouTube, AI for TikTok, AI Avatars, AI for social media, AI for agencies, AI for e-commerce, AI for education, AI for finance, AI for healthcare, AI for real estate, Prompt engineering, OpenAI updates, AI workspace, coding copilots, autonomous agents, GPT-4o, chatgpt alternatives, best free AI tools, Runway ML, LLaMA 4, AI agents framework, AI replaces [job], Mistral AI, RAG architecture, RAG models, autonomous AI agents, GPT-4.5, MoE models, GPT-4.1, OpenAI Sora, GPT-5, agentic workflows, AI content creation, LLM tooling, AI + Notion, LangChain, AutoGPT, AI automation tools, AI workflows, AI-powered coding, prompt marketplaces, smart AI agents, AI side hustles, YouTube AI tools, generative agents, AI for developers, voice cloning tools, AI script generators, AI shorts generator, AI assistants, open-source LLMs, Google Gemini, Claude AI, AI video editors, AI workflow automation, OpenAI, Anthropic, Google DeepMind, Microsoft Copilot"
        ></textarea>
      </div>

      <div class="flex items-center justify-between pt-4">
        <div class="text-sm text-gray-300">All keys are stored securely on the server</div>
        <button 
          type="submit" 
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          :disabled="loading"
        >
          {{ loading ? 'Saving...' : 'Save Settings' }}
        </button>
      </div>
    </form>

    <div v-if="error" class="mt-4 p-3 bg-red-900 text-red-200 rounded">
      {{ error }}
    </div>
    <div v-if="success" class="mt-4 p-3 bg-green-900 text-green-200 rounded">
      Settings saved successfully!
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import axios from 'axios';

interface Voice {
  voice_id: string;
  name: string;
  labels?: {
    gender?: string;
    accent?: string;
  };
}

interface Settings {
  openai_api_key: string;
  replicate_api_key: string;
  elevenlabs_api_key: string;
  openai_model: string;
  elevenlabs_voice_id: string;
  audience: string;
  flux_model: string;
}

const settings = ref<Settings>({
  flux_model: '',
  openai_api_key: '',
  replicate_api_key: '',
  elevenlabs_api_key: '',
  openai_model: '',
  elevenlabs_voice_id: '',
  audience: ''
});

const voices = ref<Voice[]>([]);
const loading = ref(false);
const error = ref('');
const success = ref(false);

// Fetch current settings
onMounted(async () => {
  try {
    const accessToken = localStorage.getItem('accessToken');
    const response = await axios.get('/api/users/api-keys/', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'X-API-KEY': import.meta.env.VITE_X_API_KEY
      }
    });
    
    if (response.data) {
      settings.value = { ...settings.value, ...response.data };
      if (settings.value.elevenlabs_api_key) {
        await fetchVoices();
      }
    }
  } catch (err) {
    error.value = 'Failed to load current settings';
  }
});

const handleElevenLabsKeyChange = async () => {
  if (settings.value.elevenlabs_api_key) {
    await fetchVoices();
  } else {
    voices.value = [];
  }
};

const fetchVoices = async () => {
  try {
    const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
      headers: {
        'xi-api-key': settings.value.elevenlabs_api_key
      }
    });
    voices.value = response.data.voices || [];
  } catch (err) {
    voices.value = [];
    console.error('Failed to fetch voices:', err);
  }
};

const saveSettings = async () => {
  loading.value = true;
  error.value = '';
  success.value = false;

  try {
    const accessToken = localStorage.getItem('accessToken');
    await axios.put('/api/users/update-keys/', settings.value, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
        'X-API-KEY': import.meta.env.VITE_X_API_KEY
      }
    });
    success.value = true;
  } catch (err) {
    error.value = 'Failed to save settings';
  } finally {
    loading.value = false;
  }
};
</script>
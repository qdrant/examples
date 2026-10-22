<template>
  <aside class="h-screen w-56 bg-gray-900 text-white flex flex-col py-8 px-4 fixed left-0 top-0 z-40">
    <div class="mb-10 text-2xl font-bold tracking-tight text-center">Vidyne</div>
    <nav class="flex flex-col gap-2">
      <router-link to="/" class="sidebar-link" active-class="bg-gray-800">Home</router-link>
      <router-link to="/create" class="sidebar-link" active-class="bg-gray-800">Create Video</router-link>
      <router-link to="/my-videos" class="sidebar-link" active-class="bg-gray-800">My Videos</router-link>
      <router-link to="/settings" class="sidebar-link" active-class="bg-gray-800">Settings</router-link>
    </nav>
    <div class="flex-1" />
    <button
      @click="handleLogout"
      class="w-full py-2 mt-4 bg-red-600 hover:bg-red-700 text-white rounded font-semibold text-sm transition"
    >
      Logout
    </button>
    <div class="text-xs text-gray-400 text-center mt-4">&copy; 2025 Vidyne</div>
  </aside>
</template>

<script setup lang="ts">
import { useAuthStore } from '../store/auth';
import { useRouter } from 'vue-router';

const auth = useAuthStore();
const router = useRouter();

function handleLogout() {
  auth.logout();
  router.push('/signin');
}
</script>

<style scoped>
.sidebar-link {
  display: block;
  padding: 0.75rem 1rem;
  border-radius: 0.375rem;
  font-weight: 500;
  color: inherit;
  transition: background 0.2s;
}
.sidebar-link:hover {
  background: #2d3748;
}
.router-link-exact-active {
  background: #2d3748;
}
</style>

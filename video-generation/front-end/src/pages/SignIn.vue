<template>
  <AuthLayout>
    <form @submit.prevent="handleLogin" class="max-w-md mx-auto w-full">
      <h2 class="text-2xl font-bold mb-8 text-white">Log in to Vidyne</h2>
      <div v-if="auth.error" class="mb-4 text-red-400">{{ auth.error }}</div>
      <div class="mb-4">
        <label class="block text-gray-300 mb-1">Email</label>
        <input v-model="email" type="email" required class="w-full px-4 py-2 rounded bg-[#181B36] text-white outline-none focus:ring-2 focus:ring-blue-400" placeholder="Enter your email..." />
      </div>
      <div class="mb-6">
        <label class="block text-gray-300 mb-1">Password</label>
        <input v-model="password" type="password" required class="w-full px-4 py-2 rounded bg-[#181B36] text-white outline-none focus:ring-2 focus:ring-blue-400" placeholder="Enter your password..." />
      </div>
      <button type="submit" class="w-full py-2 rounded bg-blue-500 hover:bg-blue-600 text-white font-semibold transition">Log in to your account</button>
      <div class="mt-6 text-center text-gray-300">
        Don't have an account?
        <router-link to="/signup" class="text-blue-400 hover:underline">Sign up here</router-link>
      </div>
    </form>
  </AuthLayout>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../store/auth';
import AuthLayout from '../components/AuthLayout.vue';

const email = ref('');
const password = ref('');
const auth = useAuthStore();
const router = useRouter();

const handleLogin = async () => {
  await auth.login(email.value, password.value);
  if (auth.isAuthenticated) {
    router.push('/');
  }
};
</script>

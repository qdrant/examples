<script setup lang="ts">
import Sidebar from './components/Sidebar.vue';
import { computed, onMounted, onUnmounted, watch } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();
const showSidebar = computed(() => !['/signin', '/signup'].includes(route.path));

function updateHomePageClass() {
  const app = document.getElementById('app');
  if (!app) return;
  if (route.path === '/') {
    app.classList.add('home-page');
  } else {
    app.classList.remove('home-page');
  }
}

onMounted(() => {
  updateHomePageClass();
});

watch(
  () => route.path,
  () => {
    updateHomePageClass();
  }
);

onUnmounted(() => {
  const app = document.getElementById('app');
  if (app) app.classList.remove('home-page');
});
</script>

<template>
  <div class="flex">
    <Sidebar v-if="showSidebar" />
    <div :class="showSidebar ? 'flex-1 ml-56' : 'flex-1'">
      <router-view />
    </div>
  </div>
</template>




<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>

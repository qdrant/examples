import { createRouter, createWebHistory } from 'vue-router';
import type { RouteRecordRaw } from 'vue-router';
import SignIn from '../pages/SignIn.vue';
import SignUp from '../pages/SignUp.vue';
import Home from '../pages/Home.vue';
import { useAuthStore } from '../store/auth';

const routes: Array<RouteRecordRaw> = [
  // ...existing routes
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('../views/Settings.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/my-videos',
    name: 'MyVideos',
    component: () => import('../views/MyVideos.vue'),
    meta: { requiresAuth: true },
  },
  { path: '/signin', name: 'SignIn', component: SignIn },
  { path: '/signup', name: 'SignUp', component: SignUp },
  { 
    path: '/oauth2callback', 
    name: 'OAuth2Callback',
    component: () => import('../views/OAuth2Callback.vue'),
    meta: { requiresAuth: true } 
  },
  {
    path: '/',
    name: 'Home',
    component: Home,
    meta: { requiresAuth: true },
  },
  {
    path: '/create',
    name: 'Create',
    component: () => import('../views/Create.vue'),
    meta: { requiresAuth: true, requiresSubscription: true },
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach(async (to, _, next) => {
  const auth = useAuthStore();
  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    next({ name: 'SignIn' });
    return;
  }
  if (to.meta.requiresSubscription) {
    const accessToken = localStorage.getItem('accessToken');
    if (!accessToken) {
      next({ name: 'SignIn' });
      return;
    }
  
  }
  next();
});

export default router;

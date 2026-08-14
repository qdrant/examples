import { defineStore } from 'pinia';
import axios from 'axios';

export const useAuthStore = defineStore('auth', {
  state: () => ({
    accessToken: localStorage.getItem('accessToken') || null,
    refreshToken: localStorage.getItem('refreshToken') || null,
    isAuthenticated: !!localStorage.getItem('accessToken'),
    loading: false,
    error: null as string | null,
  }),
  actions: {
    async login(email: string, password: string) {
      this.loading = true;
      this.error = null;
      try {
        const res = await axios.post('/api/users/login/', { email, password });
        this.accessToken = res.data.access;
        this.refreshToken = res.data.refresh;
        this.isAuthenticated = true;
        localStorage.setItem('accessToken', res.data.access);
        localStorage.setItem('refreshToken', res.data.refresh);
      } catch (err) {
        if (typeof err === 'object' && err !== null && 'response' in err) {
          this.error = (err as any).response?.data?.message || 'Login failed';
        } else {
          this.error = 'Login failed';
        }
        this.isAuthenticated = false;
      } finally {
        this.loading = false;
      }
    },
    async register(email: string, password: string) {
      this.loading = true;
      this.error = null;
      try {
        await axios.post('/api/users/register/', { email, password });
      } catch (err) {
        if (typeof err === 'object' && err !== null && 'response' in err) {
          this.error = (err as any).response?.data?.message || 'Signup failed';
        } else {
          this.error = 'Signup failed';
        }
      } finally {
        this.loading = false;
      }
    },
    logout() {
      this.accessToken = null;
      this.refreshToken = null;
      this.isAuthenticated = false;
      localStorage.removeItem('accessToken');
      localStorage.removeItem('refreshToken');
    },
  },
});

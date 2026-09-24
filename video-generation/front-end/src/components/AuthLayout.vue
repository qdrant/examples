<template>
  <!-- Full viewport, split-screen layout -->
  <div class="fixed inset-0 w-screen h-screen flex overflow-hidden">
    <!-- Left side: Branding/Testimonial -->
    <div class="hidden md:flex w-1/2 h-full items-start flex-col justify-between bg-gradient-to-br from-[#181B36] via-[#1E174B] to-[#3a1c71] text-white p-12 relative">
      <img src="../assets/logo.png" alt="Vidyne Logo" class="mb-12 mt-4 w-28" />
      <div class="flex-1 flex flex-col justify-center">
        <blockquote class="text-2xl font-medium mb-8 max-w-md">“Vidyne runs your Shorts channel on autopilot.”</blockquote>
        <div class="flex items-center gap-4 mt-6">
          <!-- <img src="/avatar.png" alt="User" class="w-12 h-12 rounded-full border-2 border-blue-400" /> -->
          <div>
            <!-- <div class="font-semibold">Jane Doe</div> -->
            <!-- <div class="text-xs text-blue-300">1.2M subs</div> -->
          </div>
        </div>
      </div>
      <div class="mb-4"></div>
    </div>
    <!-- Right side: Form -->
    <div class="flex flex-col justify-center items-center w-full md:w-1/2 h-full px-8 py-12 bg-[#232740]">
      <slot />
    </div>
  </div>
</template>

<script setup>
// No script needed
</script>

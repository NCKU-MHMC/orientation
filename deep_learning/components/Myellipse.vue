<!--

Simple Arrow

<arrow x1="10" y1="20" x2="100" y2="200" color="green" width="3" />

<arrow v-bind="{ x1:10, y1:10, x2:200, y2:200 }"/>

-->

<script setup lang="ts">
import { customAlphabet } from 'nanoid'

defineProps<{
  x: number | string
  y: number | string
  rx: number | string
  ry: number | string
  width?: number | string
  color?: string
}>()

const nanoid = customAlphabet('abcedfghijklmn', 10)

const id = nanoid()
</script>

<template>
  <svg 
    class="absolute left-0 top-0 pointer-events-none"
    :width="+x + +rx + 50"
    :height="+y + +ry + 50">
    <ellipse 
      :cx="+x" :cy="+y" :rx="+rx" :ry="+ry"
      :style="`fill:#0000;stroke:${color || 'currentColor'};stroke-width:${width || 2}`"/>
  </svg>
</template>
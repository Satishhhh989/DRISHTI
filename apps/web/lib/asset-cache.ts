/**
 * DRISHTI - Unified Asset Cache & Manager
 * Phase 08: Unified Hybrid Renderer
 *
 * Provides shared, deduplicated loading and memory-conscious caching of:
 * 1. 3D Human Meshes (.glb via GLTFLoader)
 * 2. Textures (source photograph & depth visualization via TextureLoader)
 *
 * Guarantees that switching between representations (wireframe <-> solid3d <-> articulated <-> combined)
 * does NOT re-fetch or re-parse assets from disk or network.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

class UnifiedAssetCache {
  private modelCache = new Map<string, THREE.Group>();
  private modelInFlight = new Map<string, Promise<THREE.Group>>();
  private textureCache = new Map<string, THREE.Texture>();
  private textureInFlight = new Map<string, Promise<THREE.Texture>>();
  private gltfLoader: GLTFLoader;
  private textureLoader: THREE.TextureLoader;

  constructor() {
    this.gltfLoader = new GLTFLoader();
    this.textureLoader = new THREE.TextureLoader();
  }

  /**
   * Loads or returns a cached GLTF model template, returning a cloned instance.
   */
  async loadModel(url: string): Promise<THREE.Group> {
    const cached = this.modelCache.get(url);
    if (cached) {
      return cached.clone(true);
    }

    const inFlight = this.modelInFlight.get(url);
    if (inFlight) {
      const template = await inFlight;
      return template.clone(true);
    }

    const loadPromise = new Promise<THREE.Group>((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => {
          this.modelCache.set(url, gltf.scene);
          this.modelInFlight.delete(url);
          resolve(gltf.scene);
        },
        undefined,
        (err) => {
          this.modelInFlight.delete(url);
          reject(err);
        }
      );
    });

    this.modelInFlight.set(url, loadPromise);
    const template = await loadPromise;
    return template.clone(true);
  }

  /**
   * Loads or returns a cached Three.js texture.
   */
  async loadTexture(url: string, isColor: boolean = true): Promise<THREE.Texture> {
    const cached = this.textureCache.get(url);
    if (cached) {
      return cached;
    }

    const inFlight = this.textureInFlight.get(url);
    if (inFlight) {
      return await inFlight;
    }

    const loadPromise = new Promise<THREE.Texture>((resolve, reject) => {
      this.textureLoader.load(
        url,
        (texture) => {
          if (isColor) {
            texture.colorSpace = THREE.SRGBColorSpace;
          }
          texture.minFilter = THREE.LinearFilter;
          texture.generateMipmaps = false;
          this.textureCache.set(url, texture);
          this.textureInFlight.delete(url);
          resolve(texture);
        },
        undefined,
        (err) => {
          this.textureInFlight.delete(url);
          reject(err);
        }
      );
    });

    this.textureInFlight.set(url, loadPromise);
    return await loadPromise;
  }

  /**
   * Checks whether a model URL is already loaded and cached in memory.
   */
  hasModel(url: string): boolean {
    return this.modelCache.has(url);
  }

  /**
   * Checks whether a texture URL is already loaded and cached in memory.
   */
  hasTexture(url: string): boolean {
    return this.textureCache.has(url);
  }

  /**
   * Disposes of and frees all cached textures and model geometries.
   */
  disposeAll(): void {
    // Dispose textures
    for (const texture of this.textureCache.values()) {
      texture.dispose();
    }
    this.textureCache.clear();
    this.textureInFlight.clear();

    // Dispose model geometries and materials in template cache
    for (const group of this.modelCache.values()) {
      group.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          mesh.geometry?.dispose();
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => m.dispose());
          } else {
            mesh.material?.dispose();
          }
        }
      });
    }
    this.modelCache.clear();
    this.modelInFlight.clear();
  }
}

export const assetCache = new UnifiedAssetCache();


/**
 * Helper to inspect dimensions of a chosen image file in the browser.
 */

export interface ImageDimensions {
  width: number;
  height: number;
  aspect: number;
}

export function getImageDimensions(file: File): Promise<ImageDimensions> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      const width = img.naturalWidth || img.width;
      const height = img.naturalHeight || img.height;
      URL.revokeObjectURL(url);
      resolve({
        width,
        height,
        aspect: height > 0 ? Math.round((width / height) * 1000) / 1000 : 1,
      });
    };

    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load and inspect image dimensions"));
    };

    img.src = url;
  });
}
